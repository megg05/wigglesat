import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from models.actuators import BoomArray

def plot_simulation_results(t_hist, x_hist, save_fig=False, filename="wigglesat_results.png"):
    """
    Plots state trajectory histories:
    1. Quaternion attitude
    2. Spacecraft angular velocities
    3. Boom hinge angles
    4. Boom hinge angular rates
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('WiggleSat Simulation Trajectory', fontsize=14, fontweight='bold')

    # 1. Quaternions
    axs[0].plot(t_hist, x_hist[:, 0], label='q0 (scalar)', color='black', linestyle='--')
    axs[0].plot(t_hist, x_hist[:, 1], label='q1')
    axs[0].plot(t_hist, x_hist[:, 2], label='q2')
    axs[0].plot(t_hist, x_hist[:, 3], label='q3')
    axs[0].set_ylabel('Attitude (Quat)')
    axs[0].grid(True)
    axs[0].legend(loc='upper right', ncol=4)

    # 2. Base Angular Velocity (rad/s)
    axs[1].plot(t_hist, x_hist[:, 4], label=r'$\omega_x$', color='r')
    axs[1].plot(t_hist, x_hist[:, 5], label=r'$\omega_y$', color='g')
    axs[1].plot(t_hist, x_hist[:, 6], label=r'$\omega_z$', color='b')
    axs[1].set_ylabel('Ang. Velocity [rad/s]')
    axs[1].grid(True)
    axs[1].legend(loc='upper right')

    # 3. Boom Angles (rad or deg)
    axs[2].plot(t_hist, np.rad2deg(x_hist[:, 7]), label=r'$\theta_1$', color='c')
    axs[2].plot(t_hist, np.rad2deg(x_hist[:, 8]), label=r'$\theta_2$', color='m')
    axs[2].plot(t_hist, np.rad2deg(x_hist[:, 9]), label=r'$\theta_3$', color='y')
    axs[2].set_ylabel('Boom Angle [deg]')
    axs[2].grid(True)
    axs[2].legend(loc='upper right')

    # 4. Boom Angular Rates (rad/s)
    axs[3].plot(t_hist, x_hist[:, 10], label=r'$\dot{\theta}_1$', color='c', linestyle=':')
    axs[3].plot(t_hist, x_hist[:, 11], label=r'$\dot{\theta}_2$', color='m', linestyle=':')
    axs[3].plot(t_hist, x_hist[:, 12], label=r'$\dot{\theta}_3$', color='y', linestyle=':')
    axs[3].set_ylabel('Boom Rate [rad/s]')
    axs[3].set_xlabel('Time [s]')
    axs[3].grid(True)
    axs[3].legend(loc='upper right')

    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
    plt.show()


def _quat_to_rotmat(q, body_to_inertial=True):
    """
    Rotation matrix from a scalar-first quaternion [q0, q1, q2, q3].
    Assumes q rotates body-frame vectors into the inertial frame; set
    body_to_inertial=False if core.attitude uses the opposite convention.
    """
    q = np.asarray(q, dtype=float)
    a, b, c, d = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2*(c*c + d*d), 2*(b*c - a*d),     2*(b*d + a*c)],
        [2*(b*c + a*d),     1 - 2*(b*b + d*d), 2*(c*d - a*b)],
        [2*(b*d - a*c),     2*(c*d + a*b),     1 - 2*(b*b + c*c)],
    ])
    return R if body_to_inertial else R.T


def _cube_geometry(side):
    """Vertices (8, 3) and face index lists for an axis-aligned cube centred on the origin."""
    h = side / 2.0
    verts = np.array([[sx * h, sy * h, sz * h]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    # vertex index = 4*(sx>0) + 2*(sy>0) + (sz>0)
    faces = [[0, 1, 3, 2], [4, 5, 7, 6],   # -x, +x
             [0, 1, 5, 4], [2, 3, 7, 6],   # -y, +y
             [0, 2, 6, 4], [1, 3, 7, 5]]   # -z, +z
    return verts, faces


def _unit_sphere_quads(n_lat=8, n_lon=14):
    """Quad mesh of the unit sphere as an array of shape (n_quads, 4, 3)."""
    lat = np.linspace(0, np.pi, n_lat + 1)
    lon = np.linspace(0, 2 * np.pi, n_lon + 1)
    P = np.array([[[np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)]
                   for lo in lon] for la in lat])
    quads = [[P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]]
             for i in range(n_lat) for j in range(n_lon)]
    return np.array(quads)


def animate_booms_3d(t_hist, x_hist, config, stride=5, save_gif=False,
                     cube_size=0.1, tip_radius=None, body_to_inertial=True):
    """
    Creates a 3D animation of the satellite: a cube centred on the chassis CoM,
    the 3 booms, and a sphere at the end of each boom for its tip mass.

    The whole assembly is rotated by the attitude quaternion (x_hist[:, 0:4]), so
    the cube tumbles in response to the boom reaction torques.

    cube_size        : chassis cube edge length [m] (default 0.1 puts the default hinges on the face centres)
    tip_radius       : sphere radius [m]; scalar, length-3 sequence, or None to scale with
                       tip mass (0.03 m at 0.05 kg, volume-proportional)
    body_to_inertial : quaternion convention; flip if the cube appears to rotate the wrong way
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Boom Configuration Motion', fontweight='bold')

    # Use the same boom geometry as the dynamics (hinges, lengths, boom directions, tip masses)
    booms = BoomArray(config['booms'])
    hinges = booms.r_hinges
    l_booms = booms.l_rods
    m_tips = booms.m_tips

    if tip_radius is None:
        tip_radii = 0.03 * (m_tips / 0.05) ** (1.0 / 3.0)
    else:
        tip_radii = np.broadcast_to(np.asarray(tip_radius, dtype=float), (3,))

    # Downsample history for smooth animation
    t_sub = t_hist[::stride]
    x_sub = x_hist[::stride]

    colors = ['r', 'g', 'b']

    # Chassis cube (rotates with attitude)
    cube_verts, cube_faces = _cube_geometry(cube_size)
    cube = Poly3DCollection([cube_verts[f] for f in cube_faces],
                            facecolor='lightgray', edgecolor='k', alpha=0.35, linewidths=1.2)
    ax.add_collection3d(cube)

    # Boom lines and tip-mass spheres
    lines = [ax.plot([], [], [], color=colors[i], lw=3, label=f'Boom {i+1}')[0] for i in range(3)]
    sphere_quads = _unit_sphere_quads()
    spheres = []
    for i in range(3):
        sph = Poly3DCollection(sphere_quads * tip_radii[i], facecolor=colors[i],
                               edgecolor='none', alpha=0.9)
        ax.add_collection3d(sph)
        spheres.append(sph)

    # Chassis origin point
    ax.scatter([0], [0], [0], color='black', s=40, label='Chassis CoM')

    # Set 3D boundaries
    max_len = np.max(l_booms) + np.max(np.abs(hinges)) + np.max(tip_radii) + 0.1
    ax.set_xlim([-max_len, max_len])
    ax.set_ylim([-max_len, max_len])
    ax.set_zlim([-max_len, max_len])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def update(frame):
        state = x_sub[frame]
        R_att = _quat_to_rotmat(state[:4], body_to_inertial)
        theta = state[7:10]

        # Cube: body-frame vertices rotated into the inertial frame
        v = cube_verts @ R_att.T
        cube.set_verts([v[f] for f in cube_faces])

        for i in range(3):
            hinge_pt = hinges[i]
            tip_pt = booms.geometry(i, theta[i])[1]  # tip-mass position in the body frame

            hinge_w = R_att @ hinge_pt
            tip_w = R_att @ tip_pt

            # Draw line from hinge attachment point to tip of boom
            lines[i].set_data_3d(
                [hinge_w[0], tip_w[0]],
                [hinge_w[1], tip_w[1]],
                [hinge_w[2], tip_w[2]]
            )
            spheres[i].set_verts(sphere_quads * tip_radii[i] + tip_w)
        time_text.set_text(f'Time: {t_sub[frame]:.2f} s')
        return [cube] + lines + spheres + [time_text]

    anim = FuncAnimation(fig, update, frames=len(t_sub), interval=30, blit=False)

    if save_gif:
        anim.save('wigglesat_booms.gif', writer='pillow', fps=30)
        print("Animation saved as wigglesat_booms.gif")

    plt.show()
    return anim