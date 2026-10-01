"""
Physics unit tests for WiggleSat (magnetorquers off unless stated).

Ground truth: every boom is modelled as a distributed thin rod (4-point Gauss-Legendre, exact for
the quadratic integrands here) plus a point tip mass, and quantities are summed directly:

    J(theta) = J_base + sum m (|r|^2 I - r r^T)
    H        = J_base w + sum m r x (w x r + theta_dot_i * a_i x (r - r_hinge_i))
             = J(theta) w + sum_i g_i(theta_i) theta_dot_i

This is independent of the closed-form formulas in BoomArray / WiggleSat, so the tests check
those formulas rather than restate them.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from core.sat import WiggleSat

Q_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
ZERO3 = np.zeros(3)
AXES = np.eye(3)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


# --------------------------------------------------------------------------- helpers
def make_config(r_hinges=None):
    return {
        'J': np.diag([0.05, 0.05, 0.08]),
        'booms': {
            'boom_axes': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'm_rods': [0.1, 0.1, 0.1],
            'm_tips': [0.05, 0.05, 0.05],
            'l_rods': [0.5, 0.5, 0.5],
            'r_hinges': r_hinges if r_hinges is not None else [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
        },
        'magnetorquers': {'m_max': 0.2},
        'motor_dynamics': False,  # boom input is a direct joint acceleration
    }


def make_state(w=ZERO3, theta=ZERO3, theta_dot=ZERO3):
    return np.concatenate([Q_IDENTITY, w, theta, theta_dot]).astype(float)


@pytest.fixture
def sat():
    return WiggleSat(make_config(), make_state())


def point_masses(sat, i, theta_i):
    """Boom i as [(mass, position, dr/dtheta_i)]: rod split into Gauss points, plus the tip mass."""
    B = sat.Booms
    a, r_h, l = B.axes[i], B.r_hinges[i], B.l_rods[i]
    d = Rotation.from_rotvec(theta_i * a).apply(B.boom_dirs[i])
    pts = [(0.5 * B.m_rods[i] * wk, r_h + 0.5 * l * (xk + 1.0) * d) for xk, wk in zip(_GL_X, _GL_W)]
    pts.append((B.m_tips[i], r_h + l * d))
    return [(m, r, np.cross(a, r - r_h)) for m, r in pts]


def ref_inertia(sat, theta):
    J = sat.J_base.copy()
    for i in range(3):
        for m, r, _ in point_masses(sat, i, theta[i]):
            J += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return J


def ref_g(sat, i, theta_i):
    """Angular momentum about the chassis CoM per unit hinge rate of boom i (chassis fixed)."""
    return sum(m * np.cross(r, v) for m, r, v in point_masses(sat, i, theta_i))


def ref_g_rate(sat, i, theta_i, eps=1e-6):
    return (ref_g(sat, i, theta_i + eps) - ref_g(sat, i, theta_i - eps)) / (2 * eps)


def ref_momentum(sat, x):
    w, theta, theta_dot = x[4:7], x[7:10], x[10:13]
    H = ref_inertia(sat, theta) @ w
    for i in range(3):
        H = H + theta_dot[i] * ref_g(sat, i, theta[i])
    return H


def integrate(sat, x0, u_boom, u_mag=ZERO3, B_body=ZERO3, T=10.0, n=60):
    """Integrate state_derivative with tight tolerances so drift is physics, not solver error."""
    t_eval = np.linspace(0.0, T, n)
    sol = solve_ivp(
        lambda t, y: sat.state_derivative(t, y, np.asarray(u_boom, float), u_mag, B_body),
        (0.0, T), x0, t_eval=t_eval, rtol=1e-11, atol=1e-13,
    )
    assert sol.success
    return sol.t, sol.y.T


# --------------------------------------------------------------------------- 0) model geometry
def test_boom_directions_are_unit_and_perpendicular_to_hinge(sat):
    d, a = sat.Booms.boom_dirs, sat.Booms.axes
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(d * a, axis=1), 0.0, atol=1e-14)


@pytest.mark.parametrize("i", range(3))
def test_every_boom_changes_inertia_when_rotated(sat, i):
    """A boom lying along its own hinge axis would be inert; none may be."""
    J0 = sat.update_inertia(ZERO3).copy()
    theta = np.zeros(3)
    theta[i] = 0.5
    assert np.abs(sat.update_inertia(theta) - J0).max() > 1e-3


def test_inertia_matches_distributed_mass_model(sat):
    for theta in ([0, 0, 0], [0.3, -0.5, 0.8], [2.0, 1.1, -2.5]):
        np.testing.assert_allclose(sat.update_inertia(np.array(theta, float)),
                                   ref_inertia(sat, np.array(theta, float)), atol=1e-12)


def test_inertia_rate_matches_finite_difference(sat):
    theta, theta_dot, eps = np.array([0.3, -0.5, 0.8]), np.array([1.0, -2.0, 0.7]), 1e-6
    sat.update_inertia(theta, theta_dot)
    J_dot = sat.J_dot.copy()
    fd = (ref_inertia(sat, theta + eps * theta_dot) - ref_inertia(sat, theta - eps * theta_dot)) / (2 * eps)
    np.testing.assert_allclose(J_dot, fd, atol=1e-8)


@pytest.mark.parametrize("i", range(3))
def test_boom_momentum_vector_and_rate_match_distributed_mass_model(sat, i):
    for th in (0.0, 0.7, -1.9):
        np.testing.assert_allclose(sat.Booms.momentum_vector(i, th), ref_g(sat, i, th), atol=1e-12)
        np.testing.assert_allclose(sat.Booms.momentum_vector_rate(i, th), ref_g_rate(sat, i, th), atol=1e-8)


# --------------------------------------------------------------------------- 1) angular momentum
BOOM_ACCELS = {
    "boom1_only": [0.5, 0.0, 0.0],
    "boom2_only": [0.0, 0.5, 0.0],
    "boom3_only": [0.0, 0.0, 0.5],
    "all_booms": [0.5, -0.3, 0.1],
}


@pytest.mark.parametrize("u_boom", BOOM_ACCELS.values(), ids=BOOM_ACCELS.keys())
def test_momentum_stays_zero_from_rest(sat, u_boom):
    """Starting from rest, H = 0 must hold for all time, even though the booms (and body) move."""
    _, X = integrate(sat, make_state(), u_boom)

    H = np.array([ref_momentum(sat, x) for x in X])
    assert np.abs(H).max() < 1e-9

    # Non-vacuous: the system really did move, and the body really did react.
    assert np.abs(X[-1, 10:13]).max() > 1.0
    assert np.linalg.norm(X[-1, 4:7]) > 1e-3


@pytest.mark.parametrize("u_boom", BOOM_ACCELS.values(), ids=BOOM_ACCELS.keys())
def test_momentum_magnitude_conserved_with_initial_spin(sat, u_boom):
    """With initial body spin and moving booms, |H| is constant (H itself rotates in the body frame)."""
    x0 = make_state(w=np.array([0.3, -0.2, 0.4]), theta=np.array([0.1, -0.2, 0.3]),
                    theta_dot=np.array([0.2, 0.1, -0.1]))
    _, X = integrate(sat, x0, u_boom)

    H_norm = np.array([np.linalg.norm(ref_momentum(sat, x)) for x in X])
    assert H_norm[0] > 0.05  # meaningful nonzero momentum to conserve
    assert np.ptp(H_norm) < 1e-9


def test_model_angular_momentum_matches_ground_truth(sat):
    x = make_state(w=np.array([0.3, -0.2, 0.4]), theta=np.array([0.1, -0.2, 0.3]),
                   theta_dot=np.array([0.2, 0.1, -0.1]))
    np.testing.assert_allclose(sat.angular_momentum(x), ref_momentum(sat, x), atol=1e-12)


def test_momentum_conserved_with_zero_dipole_in_magnetic_field(sat):
    """A magnetic field alone does nothing if the magnetorquers command zero dipole."""
    x0 = make_state(w=np.array([0.1, 0.2, -0.1]))
    _, X = integrate(sat, x0, [0.5, -0.3, 0.1], u_mag=ZERO3, B_body=np.array([2e-5, -3e-5, 4e-5]))

    H_norm = np.array([np.linalg.norm(ref_momentum(sat, x)) for x in X])
    assert np.ptp(H_norm) < 1e-9


def test_magnetorquers_do_change_momentum(sat):
    """Control for the tests above: an external torque (m x B) must break conservation."""
    _, X = integrate(sat, make_state(), ZERO3, u_mag=np.array([0.1, 0.0, 0.0]),
                     B_body=np.array([0.0, 0.0, 5e-5]), T=5.0)

    # tau = m x B = [0.1,0,0] x [0,0,5e-5] = [0, -5e-6, 0]  ->  H = tau * T
    np.testing.assert_allclose(ref_momentum(sat, X[-1]), [0.0, -5e-6 * 5.0, 0.0], rtol=1e-6, atol=1e-9)


def test_simulator_conserves_momentum_with_default_tolerances():
    """End to end through Simulator.run: correct time axis and momentum held with its default RK45 settings."""
    from simulation import Simulator
    from control.controllers import DirectInputController

    sim = Simulator(make_config(), DirectInputController(np.array([0.5, -0.3, 0.1]), ZERO3), make_state())
    t, X = sim.run(sim_time=1.0, dt=0.01)

    np.testing.assert_allclose(np.diff(t), 0.01, atol=1e-12)
    H = np.array([ref_momentum(sim.sat, x) for x in X])
    assert np.abs(H).max() < 1e-6


# --------------------------------------------------------------------------- 2) single-boom body torque
@pytest.mark.parametrize("i", range(3))
def test_reaction_torque_from_rest_is_minus_g_alpha(sat, i):
    """
    Boom i accelerating at alpha with body and boom at rest:   tau_body = -g_i * alpha
    where g_i is the boom's momentum per unit hinge rate. Its component along the hinge is
    exactly J_boom (the hinge offset in this config is along the axis); the hinge offset adds a
    small off-axis part.
    """
    alpha = 0.7
    alphas = np.zeros(3)
    alphas[i] = alpha

    _, tau = sat.Booms.compute_reaction_torque(ZERO3, ZERO3, alphas, ZERO3, use_motor=False)

    np.testing.assert_allclose(tau, -alpha * ref_g(sat, i, 0.0), atol=1e-14)
    assert tau @ AXES[i] == pytest.approx(-sat.Booms.inertias[i] * alpha, rel=1e-12)
    assert np.linalg.norm(tau - (tau @ AXES[i]) * AXES[i]) > 1e-4   # off-axis part is really there


@pytest.mark.parametrize("i", range(3))
def test_reaction_torque_is_minus_Jb_alpha_axis_when_hinge_at_com(i):
    """Hinge at the chassis CoM: the relation collapses to tau = -J_boom * alpha * a_i, purely along the axis."""
    sat0 = WiggleSat(make_config(r_hinges=np.zeros((3, 3))), make_state())
    alphas = np.zeros(3)
    alphas[i] = 0.7

    _, tau = sat0.Booms.compute_reaction_torque(ZERO3, ZERO3, alphas, ZERO3, use_motor=False)

    np.testing.assert_allclose(tau, -sat0.Booms.inertias[i] * 0.7 * AXES[i], atol=1e-15)


@pytest.mark.parametrize("i", range(3))
def test_reaction_torque_is_linear_in_acceleration(sat, i):
    """Doubling alpha doubles the torque; reversing alpha reverses it."""
    def tau_for(alpha):
        a = np.zeros(3)
        a[i] = alpha
        return sat.Booms.compute_reaction_torque(ZERO3, ZERO3, a, ZERO3, use_motor=False)[1]

    np.testing.assert_allclose(tau_for(2.0), 2.0 * tau_for(1.0), atol=1e-15)
    np.testing.assert_allclose(tau_for(-1.0), -tau_for(1.0), atol=1e-15)


@pytest.mark.parametrize("i", range(3))
def test_reaction_torque_general_single_boom(sat, i):
    """
    Boom i at angle theta with rate theta_dot and acceleration alpha, chassis spinning at w:
        tau_body = -( g_i*alpha + g_i'(theta)*theta_dot^2 + theta_dot * (w x g_i) )
    """
    w = np.array([0.4, -0.7, 0.2])
    theta_i, theta_dot_i, alpha = 0.6, 1.3, 0.5
    theta, theta_dot, alphas = np.zeros(3), np.zeros(3), np.zeros(3)
    theta[i], theta_dot[i], alphas[i] = theta_i, theta_dot_i, alpha

    _, tau = sat.Booms.compute_reaction_torque(theta, theta_dot, alphas, w, use_motor=False)

    g, g_rate = ref_g(sat, i, theta_i), ref_g_rate(sat, i, theta_i)
    expected = -(g * alpha + g_rate * theta_dot_i**2 + theta_dot_i * np.cross(w, g))
    np.testing.assert_allclose(tau, expected, atol=1e-8)


@pytest.mark.parametrize("i", range(3))
def test_body_angular_acceleration_from_rest(sat, i):
    """
    From rest at theta = 0 the body responds with   w_dot = -J^-1 * g_i * alpha
    (and the boom itself gets theta_ddot = alpha).
    """
    alpha = 0.4
    u = np.zeros(3)
    u[i] = alpha

    dx = sat.state_derivative(0.0, make_state(), u, ZERO3)

    expected_w_dot = -alpha * np.linalg.solve(ref_inertia(sat, ZERO3), ref_g(sat, i, 0.0))
    np.testing.assert_allclose(dx[4:7], expected_w_dot, atol=1e-12)
    assert dx[10 + i] == pytest.approx(alpha)
    np.testing.assert_allclose(np.delete(dx[10:13], i), 0.0, atol=1e-15)


@pytest.mark.parametrize("i", range(3))
def test_body_rate_follows_boom_rate(sat, i):
    """
    Single boom accelerating from rest: H = 0 at every instant gives
        w(t) = -J(theta(t))^-1 * g_i(theta(t)) * theta_dot(t),
    with theta_dot = alpha * t and theta = alpha * t^2 / 2.
    """
    alpha = 0.5
    u = np.zeros(3)
    u[i] = alpha
    t, X = integrate(sat, make_state(), u, T=6.0)

    for tk, x in zip(t, X):
        theta = np.zeros(3)
        theta[i] = 0.5 * alpha * tk**2
        expected_w = -(alpha * tk) * np.linalg.solve(ref_inertia(sat, theta), ref_g(sat, i, theta[i]))
        np.testing.assert_allclose(x[4:7], expected_w, atol=1e-8)