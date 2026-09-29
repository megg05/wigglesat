import numpy as np

class BoomArray:
    """3 actuated 1-DOF booms mounted on the satellite chassis."""
    def __init__(self, config):
        self.axes = np.array([
            axis / np.linalg.norm(axis) for axis in config['boom_axes']
        ])  # (3, 3) normalized hinge unit vectors
        
        self.inertias = np.array(config['boom_inertias'])  # (3,) moments of inertia about hinges
        self.m_booms = np.array(config.get('m_booms', [0.1, 0.1, 0.1]))  # kg
        self.l_booms = np.array(config.get('l_booms', [0.5, 0.5, 0.5]))  # m
        self.r_hinges = np.array(config.get('r_hinges', [
            [0.05, 0.0, 0.0], 
            [0.0, 0.05, 0.0], 
            [0.0, 0.0, 0.05]
        ]))  # (3, 3) hinge attachment vectors in base body frame

    def _get_boom_rotation(self, i, theta_i):
        """Computes Rodrigues' rotation matrix for boom i by angle theta around its hinge axis."""
        a = self.axes[i]
        cos_t, sin_t = np.cos(theta_i), np.sin(theta_i)
        K = np.array([
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0]
        ])
        return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)

    def compute_reaction_torque(self, theta, theta_dot, alpha_booms, w_base):
        """
        Computes dynamic reaction torque and Coriolis torque acting on chassis from boom movement.
        """
        tau_reaction = np.zeros(3)
        for i in range(3):
            axis_i = self.axes[i]
            J_boom = self.inertias[i]
            
            # Primary reaction torque from boom joint angular acceleration
            tau_reaction -= J_boom * alpha_booms[i] * axis_i
            
            # Gyroscopic/Coriolis torque coupling boom joint rate and base angular velocity
            tau_reaction -= J_boom * np.cross(w_base, theta_dot[i] * axis_i)

        return tau_reaction


class Magnetorquer:
    """3-axis magnetic torque rods for momentum desaturation."""
    def __init__(self, config):
        self.m_max = config.get('m_max', 0.2)  # Maximum magnetic dipole moment (A*m^2)

    def compute_torque(self, m_cmd, B_body):
        """
        Computes torque from magnetic dipole moment m in local magnetic field B: tau = m x B
        """
        m_clamped = np.clip(m_cmd, -self.m_max, self.m_max)
        return np.cross(m_clamped, B_body)