import numpy as np

def hat(v):
    """Returns skew-symmetric matrix for a 3-vector v."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def L_mult(q):
    """Left quaternion multiplication matrix for q = [s, v]."""
    s, v = q[0], q[1:]
    L = np.zeros((4, 4))
    L[0, 0] = s
    L[0, 1:] = -v
    L[1:, 0] = v
    L[1:, 1:] = s * np.eye(3) + hat(v)
    return L

def quat_mult(q1, q2):
    """Multiply two unit quaternions q1 * q2."""
    return L_mult(q1) @ q2

def quat_inverse(q):
    """Returns conjugate/inverse of unit quaternion q."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_kinematics(q, w):
    """Quaternion time derivative: dq/dt = 0.5 * q x [0, w]."""
    omega_q = np.array([0.0, w[0], w[1], w[2]])
    return 0.5 * L_mult(q) @ omega_q

def delta_q_from_phi(phi):
    """Convert small error vector phi (3x1) to error quaternion dq."""
    s = np.sqrt(1.0 - 0.25 * np.dot(phi, phi))
    return np.array([s, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]])