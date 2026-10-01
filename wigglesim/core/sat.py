import numpy as np
from core.attitude import quat_kinematics
from models.actuators import BoomArray, Magnetorquer

class WiggleSat:
    def __init__(self, config, init_state):
        self.J_base = config['J']
        self.Booms = BoomArray(config['booms'])
        self.Magnetorquers = Magnetorquer(config['magnetorquers'])

        self.controller = None
        self.init_state = init_state
        self.state = init_state

        self.J = self.J_base.copy()
        self.J_inv = np.linalg.inv(self.J)
        self.J_dot = np.zeros((3, 3))

        self.use_motor = config['motor_dynamics']

    def update_inertia(self, theta, theta_dot=None):
        """
        Updates the inertia matrix based on the current boom angles, in base body frame.
        If theta_dot is given, also updates self.J_dot (time derivative of J in the body frame).
        """
        B = self.Booms
        J_total = self.J_base.copy()
        J_dot_total = np.zeros((3, 3))
        for i in range(3):
            r_com, r_tip, J_rod_body = B.geometry(i, theta[i])
            J_rod_parallel = B.m_rods[i] * (np.dot(r_com, r_com) * np.eye(3) - np.outer(r_com, r_com))
            J_tip_parallel = B.m_tips[i] * (np.dot(r_tip, r_tip) * np.eye(3) - np.outer(r_tip, r_tip))

            J_total += J_rod_body + J_rod_parallel + J_tip_parallel

            if theta_dot is not None:
                # Each boom point mass moves as r = r_hinge + R p, so r_dot = theta_dot * (a x (r - r_hinge)).
                a = B.axes[i]
                a_hat = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
                J_dot_total += theta_dot[i] * (a_hat @ J_rod_body - J_rod_body @ a_hat)  # rotating rod-about-own-COM term
                for m, r in ((B.m_rods[i], r_com), (B.m_tips[i], r_tip)):
                    r_dot = theta_dot[i] * np.cross(a, r - B.r_hinges[i])
                    J_dot_total += m * (2.0 * np.dot(r, r_dot) * np.eye(3) - np.outer(r_dot, r) - np.outer(r, r_dot))

        self.J = J_total
        self.J_dot = J_dot_total
        self.J_inv = np.linalg.inv(J_total)
        return J_total

    def angular_momentum(self, state):
        """
        Total angular momentum about the chassis CoM, in the body frame:
            H = J(theta) w + sum_i g_i(theta_i) * theta_dot_i
        With no external torque this is conserved in the inertial frame (so |H| is constant).
        """
        w, theta, theta_dot = state[4:7], state[7:10], state[10:13]
        self.update_inertia(theta)
        H = self.J @ w
        for i in range(3):
            H = H + self.Booms.momentum_vector(i, theta[i]) * theta_dot[i]
        return H

    def state_derivative(self, t, state, u_boom, u_mag, B_body=np.array([0.0,0.0,0.0])):
        """
        Computes ODE derivatives.
        state: [q(4), w(3), theta_booms(3), thetadot_booms(3)]
        control inputs: 
            u_boom (3 boom accelerations)
            u_mag (3 dipole moments)

        ignoring orbit dynamics for now
        """
        q = state[:4] / np.linalg.norm(state[:4])
        w = state[4:7]
        theta = state[7:10]
        theta_dot = state[10:13]

        self.update_inertia(theta, theta_dot)

        theta_ddot,tau_reaction = self.Booms.compute_reaction_torque(theta, theta_dot, u_boom, w, self.use_motor)
        tau_mag = self.Magnetorquers.compute_torque(u_mag, B_body) # placeholder for B_body for now
        
        tau_total = tau_reaction + tau_mag

        # Rotational Dynamics (Euler's multi-body equations)
        dw_dt = self.J_inv @ (tau_total - np.cross(w, self.J @ w) - self.J_dot @ w)  # J_dot term: J varies with boom angle
        dq_dt = quat_kinematics(q, w)


        return np.concatenate([dq_dt, dw_dt, theta_dot, theta_ddot])