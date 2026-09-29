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

    def update_inertia(self, theta):
        """
        Updates the inertia matrix based on the current boom angles, in base body frame.
        """

        J_total = self.J.copy()
        for i in range(3):
            R_i = self.Booms._get_boom_rotation(i, theta[i])
            l_half = self.Booms.l_booms[i] / 2.0

            r_com = self.Booms.r_hinges[i] + R_i @ np.array([l_half, 0.0, 0.0]) # com vector in body frame

            J_local = np.diag([ # local inertia about its own com, in boom frame
                0.0,
                (1/12) * self.Booms.m_booms[i] * (self.Booms.l_booms[i] ** 2),
                (1/12) * self.Booms.m_booms[i] * (self.Booms.l_booms[i] ** 2)
            ])
            J_com_body = R_i @ J_local @ R_i.T

            r_sq = np.dot(r_com, r_com) # parallel axis theorem...
            J_parallel = self.Booms.m_booms[i] * (r_sq * np.eye(3) - np.outer(r_com, r_com))

            J_total += J_com_body + J_parallel

        self.J = J_total
        self.J_inv = np.linalg.inv(J_total)
        return J_total

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

        self.update_inertia(theta)

        tau_reaction = self.Booms.compute_reaction_torque(theta, theta_dot, u_boom, w)
        tau_mag = self.Magnetorquers.compute_torque(u_mag, B_body) # placeholder for B_body for now
        
        tau_total = tau_reaction + tau_mag

        # Rotational Dynamics (Euler's multi-body equations)
        dw_dt = self.J_inv @ (tau_total - np.cross(w, self.J @ w))
        dq_dt = quat_kinematics(q, w)

        # Joint dynamics
        dtheta_dt = theta_dot
        dtheta_dot_dt = np.array(u_boom)  # Controlled angular acceleration

        return np.concatenate([dq_dt, dw_dt, dtheta_dt, dtheta_dot_dt])