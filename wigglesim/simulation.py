import numpy as np
from scipy.integrate import solve_ivp
from core.sat import WiggleSat
from control.controllers import DirectInputController
from utils.visualization import plot_simulation_results, animate_booms_3d

class Simulator:
    def __init__(self, config, controller, init_state, dt = 0.01):
        self.dt = dt
        self.sat = WiggleSat(config, init_state)
        self.sat.controller = controller

    def run(self, sim_time=10.0, dt=0.01, rtol=1e-3, atol=1e-6):
        """
        dt is the control/logging interval; rtol/atol are the RK45 tolerances (solve_ivp's defaults).
        Each solve only spans one dt, so the defaults already hold angular momentum to ~1e-12;
        tighten them for stiffer dynamics (e.g. motor_dynamics=True) at the cost of runtime.
        """
        steps = int(sim_time / dt)
        time_history = np.arange(steps) * dt
        state_history = []

        print("starting wiggle...")
        for step in range(steps):
            t = step * dt
            state_history.append(self.sat.state.copy())

            u_boom, u_mag = self.sat.controller.u(t, self.sat.state)

            sol = solve_ivp(
                fun=lambda t_curr, y: self.sat.state_derivative(t_curr, y, u_boom, u_mag),
                t_span=(t, t + dt),
                y0=self.sat.state,
                method='RK45',
                rtol=rtol,
                atol=atol
            )
            self.sat.state = sol.y[:, -1]

        print("done wiggling. final state:", self.sat.state)
        return time_history, np.array(state_history)

if __name__ == "__main__":

    init_state = np.array([
        1.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    config = {
        'J': np.diag([0.05, 0.05, 0.08]),
        'booms': {
            'boom_axes': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'm_rods': [0.1, 0.1, 0.1],
            'm_tips': [0.05, 0.05, 0.05],
            'l_rods': [0.5, 0.5, 0.5],
            'r_hinges': [
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05]
            ],
            'motor_config': {
            'N_r': 50,             # 50 pole pairs (1.8 deg step angle)
            'T_holding': 0.15,     # Peak holding torque [N*m]
            'T_cog': 0.008,        # Cogging torque amplitude [N*m]
            'N_cog': 200,          # Cogging frequency multiplier
            'B_m': 1e-3,           # Bearing viscous friction [N*m*s/rad]
            'T_coulomb': 0.002     # Coulomb dry friction [N*m]
            }
        },
        'magnetorquers': {
            'm_max': 0.2
        },
        'motor_dynamics': False
    }

    controller = DirectInputController(
        np.array([0.5, -0.3, 0.1]), # boom accel input
        np.array([0.0, 0.0, 0.0]) # magnetorquer dipole input
    )

    simulator = Simulator(config=config, controller=controller, init_state=init_state)
    t_hist, x_hist = simulator.run(sim_time=10.0, dt=0.01)

    plot_simulation_results(t_hist, x_hist, save_fig=False)
    animate_booms_3d(t_hist, x_hist, config, stride=3, save_gif=False)