import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


def animate_booms_3d(t_hist, x_hist, config, stride=5, save_gif=False):
    """
    Creates a 3D animation showing the motion of the 3 satellite booms in time.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Boom Configuration Motion', fontweight='bold')

    # Retrieve boom geometry specs
    hinges = np.array(config['booms']['r_hinges'])
    l_booms = config['booms']['l_booms']
    boom_axes = np.array(config['booms']['boom_axes'])

    # Downsample history for smooth animation
    t_sub = t_hist[::stride]
    x_sub = x_hist[::stride]

    def _get_boom_rotation(i, theta_i):
        a = boom_axes[i] / np.linalg.norm(boom_axes[i])
        cos_t, sin_t = np.cos(theta_i), np.sin(theta_i)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)

    # Compute a local reference vector orthogonal to each hinge axis
    boom_ref_dirs = []
    for axis in boom_axes:
        a = axis / np.linalg.norm(axis)
        # Choose a reference vector not parallel to 'a'
        ref = np.array([0.0, 1.0, 0.0]) if np.allclose(a, [1, 0, 0]) else np.array([1.0, 0.0, 0.0])
        # Project and normalize to make it orthogonal to axis 'a'
        ortho = ref - np.dot(ref, a) * a
        boom_ref_dirs.append(ortho / np.linalg.norm(ortho))

    # Initialize 3D lines for booms
    colors = ['r', 'g', 'b']
    lines = [ax.plot([], [], [], color=colors[i], lw=3, label=f'Boom {i+1}')[0] for i in range(3)]
    
    # Chassis origin point
    ax.scatter([0], [0], [0], color='black', s=100, label='Chassis CoM')

    # Set 3D boundaries
    max_len = np.max(l_booms) + np.max(np.abs(hinges)) + 0.1
    ax.set_xlim([-max_len, max_len])
    ax.set_ylim([-max_len, max_len])
    ax.set_zlim([-max_len, max_len])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def update(frame):
        state = x_sub[frame]
        theta = state[7:10]
        
        for i in range(3):
            R_i = _get_boom_rotation(i, theta[i])
            hinge_pt = hinges[i]
            
            # Use the orthogonal reference direction for the unrotated boom body
            r_boom_unrotated = boom_ref_dirs[i] * l_booms[i]
            tip_pt = hinge_pt + R_i @ r_boom_unrotated

            # Draw line from hinge attachment point to tip of boom
            lines[i].set_data_3d(
                [hinge_pt[0], tip_pt[0]],
                [hinge_pt[1], tip_pt[1]],
                [hinge_pt[2], tip_pt[2]]
            )
        time_text.set_text(f'Time: {t_sub[frame]:.2f} s')
        return lines + [time_text]

    anim = FuncAnimation(fig, update, frames=len(t_sub), interval=30, blit=False)
    
    if save_gif:
        anim.save('wigglesat_booms.gif', writer='pillow', fps=30)
        print("Animation saved as wigglesat_booms.gif")

    plt.show()