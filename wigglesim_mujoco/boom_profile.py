"""python -m wigglesim_mujoco.boom_profile
run from wigglesat root"""

from dataclasses import replace
from pathlib import Path

import mujoco

import numpy as np

from mjorbit import MjoSpec, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH

from wigglesim_mujoco.actuators.motor_presets import MOTOR_PRESETS
from wigglesim_mujoco.actuators.stepper import StepperBank, configure_stepper_hinges
from wigglesim_mujoco.actuators.stepper_dcmotor import DCStepperBank, configure_dcmotors

#### sim settings #############################3
DT = 0.0001
ALT_KM = 400.0
T_END = 20.0
LOG_EVERY = 5

MODE = "position"         # "position": open-loop steps of gearbox-output angle (rad)
                          # "torque"  : closed-loop commutation of gearbox-output torque (N*m)
SAVE_DIR = Path(__file__).resolve().parents[1] / "plots"
SAVE_LOG = Path(__file__).resolve().parents[1] / "log.npz"
SHOW_PLOTS = False

#### smooth boom command profile ###########################
BOOM_ANGLE_DEG = 20.0     # command amplitude, position mode
BOOM_TORQUE = 6.0         # command amplitude, torque mode
BOOM_FREQ_HZ = 0.2
RAMP_S = 5.0

COMPLIANT = False # for motor mounts
MOTOR_PART = "5718L-01P-HP50"
MOTOR = MOTOR_PRESETS[MOTOR_PART]
MOTOR_BACKEND = "dcmotor"  # "two_phase" or "dcmotor"
ACTUATOR_NAMES = ("act_x", "act_y", "act_z")
COGGING = True
DETENT_MNM = MOTOR.detent_torque * 1e3  # peak motor-shaft detent torque (mN*m)
MICROSTEPS = MOTOR.microsteps
TUMBLE_DEG_S = [0.0, 0.0, 0.0]   # initial body rate (deg/s) to test detumbling

def smoothstep(x):
    """C2 ramp 0 -> 1. A linear ramp has a velocity step at its ends, which a stiff, lightly
    damped stepper cannot follow: it overshoots ~100 deg electrical and rings."""
    x = min(max(x, 0.0), 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def controller(t):
    """Gearbox-output command. Replace with your attitude controller.

    position mode -> shaft angles (rad); torque mode -> shaft torques (N*m).
    With compliance, actual boom angle relative to the bus includes motor mount twist.
    Keep position commands smooth (no velocity steps) or the stepper will ring / lose steps.
    """
    ramp = smoothstep(t / RAMP_S)
    amp = BOOM_TORQUE if MODE == "torque" else np.deg2rad(BOOM_ANGLE_DEG)
    return amp * ramp * np.sin(2 * np.pi * BOOM_FREQ_HZ * t + np.array([0.0, 2.0, 4.0]))


class BoomSimulation:
    """Shared model, initial state, and motor control for the headless run and viewer."""

    def __init__(self):
        xml_name = "sat_booms_compliant.xml" if COMPLIANT else "sat_booms.xml"
        xml_path = Path(__file__).with_name(xml_name)
        mj_spec = mujoco.MjSpec.from_file(str(xml_path))
        self.backend = MOTOR_BACKEND
        self.motor = replace(MOTOR, detent_torque=DETENT_MNM * 1e-3)
        if self.backend == "dcmotor":
            configure_dcmotors(mj_spec, self.motor, ACTUATOR_NAMES, cogging=COGGING)
        elif self.backend == "two_phase":
            configure_stepper_hinges(mj_spec, self.motor, ACTUATOR_NAMES)
        else:
            raise ValueError("MOTOR_BACKEND must be 'two_phase' or 'dcmotor'")
        mj_spec.option.timestep = DT
        layout = mj_spec.compile()

        suffix = "motor" if COMPLIANT else "arm"
        shaft_joints = [layout.joint(f"{axis}_{suffix}_joint").id for axis in "xyz"]
        self.shaft_q = layout.jnt_qposadr[shaft_joints]
        self.shaft_v = layout.jnt_dofadr[shaft_joints]
        if COMPLIANT:
            flex_joints = [layout.joint(f"{axis}_mount_joint").id for axis in "xyz"]
            self.flex_q = layout.jnt_qposadr[flex_joints]
            self.flex_v = layout.jnt_dofadr[flex_joints]
            self.flex_stiffness = layout.jnt_stiffness[flex_joints]
            self.flex_damping = layout.dof_damping[self.flex_v]

        spec = MjoSpec.from_mj_spec(mj_spec)
        spec.mjorbit.use_gravity_gradient = True
        spec.mjorbit.use_j2 = False
        spec.mjorbit.use_drag = False
        spec.mjorbit.use_srp = False
        spec.mjorbit.use_magnetic = False
        self.model = spec.compile(mj_timestep=DT)

        radius = R_EARTH + ALT_KM
        orbit = OrbitInit(R_eci=[radius, 0.0, 0.0],
                          V_eci=[0.0, np.sqrt(GM_EARTH / radius), 0.0])
        self.data = self.model.make_data(orbit=orbit)
        self.data.qpos[:3] = 0.0
        self.data.qvel[3:6] = np.deg2rad(TUMBLE_DEG_S)
        mjo_forward(self.model, self.data)

        if self.backend == "dcmotor":
            self.motors = DCStepperBank(layout, self.motor, ACTUATOR_NAMES,
                                        mode=MODE, microsteps=MICROSTEPS)
            self.motors.reset(self.data)
        else:
            self.motors = StepperBank(self.motor, len(ACTUATOR_NAMES), mode=MODE,
                                      cogging=COGGING, microsteps=MICROSTEPS)
            self.motors.reset(self.data.qpos[self.shaft_q])
        self.cogging_torque = np.zeros(len(ACTUATOR_NAMES))

    def apply_control(self):
        """Write controls before advancing physics and return the profile command."""
        data = self.data
        cmd = controller(data.time)
        if self.backend == "dcmotor":
            self.cogging_torque = self.motors.cogging_torque(data)
            self.motors.step(data, cmd)
        else:
            tau, self.cogging_torque, self.drive_torque = self.motors.step(
                DT, data.qpos[self.shaft_q], data.qvel[self.shaft_v], cmd)
            limit = self.motor.emergency_output_torque
            data.ctrl[:] = np.clip(tau, -limit, limit)
        return cmd

    def motor_torques(self):
        """Return total and electromagnetic output torque for the completed step."""
        if self.backend == "dcmotor":
            total = self.data.qfrc_actuator[self.shaft_v]
            return total, total - self.cogging_torque
        return self.data.ctrl[:], self.drive_torque

    def winding_power(self):
        if self.backend == "dcmotor":
            return self.motors.winding_power(self.data)
        return self.motors.winding_power()

    def boom_angles(self):
        """Boom angles relative to the bus, including motor mount twist."""
        shaft = self.data.qpos[self.shaft_q]
        return shaft + self.data.qpos[self.flex_q] if COMPLIANT else shaft


# ---- 6. Plots ---------------------------------------------------------------
def quat_angle_deg(q, q0):
    """Rotation angle (deg) of q relative to q0, for (N,4) wxyz arrays (accurate for small angles)."""
    w1, x1, y1, z1 = (q0 * np.array([1, -1, -1, -1])).T
    w2, x2, y2, z2 = q.T
    rel = np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=1)
    return np.degrees(2 * np.arctan2(np.linalg.norm(rel[:, 1:], axis=1), np.abs(rel[:, 0])))


def psd(x, fs):
    """One-sided Hann-windowed periodogram of each column of x."""
    x = x - x.mean(axis=0)
    w = np.hanning(len(x))[:, None]
    X = np.fft.rfft(x * w, axis=0)
    p = 2.0 * np.abs(X) ** 2 / (fs * np.sum(w ** 2))
    return np.fft.rfftfreq(len(x), 1.0 / fs), p


def make_plots(L):
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(SAVE_DIR).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    t = L["t"]
    ax_names = ["x arm", "y arm", "z arm"]
    figs = {}

    def new_fig(name, nrows, title, height=2.3):
        fig, axs = plt.subplots(nrows, 1, sharex=True, figsize=(10, height * nrows + 0.8))
        axs = np.atleast_1d(axs)
        axs[0].set_title(title)
        figs[name] = fig
        return fig, axs

    def finish(axs):
        for a in axs:
            a.grid(alpha=0.3)
            a.legend(loc="upper right", fontsize=8, ncol=3)
        axs[-1].set_xlabel("sim time (s)")

    # (1) boom command vs response
    fig, ax = new_fig("1_boom_response", 3, f"Boom response ({MODE} mode)")
    for j in range(3):
        if MODE == "position":
            ax[0].plot(t, np.degrees(L["cmd"][:, j]), "--", lw=1, alpha=0.7)
        ax[0].plot(t, np.degrees(L["boom_angle"][:, j]), label=ax_names[j])
        ax[1].plot(t, L["boom_rate"][:, j], label=ax_names[j])
        ax[2].plot(t, L["tau"][:, j], label=ax_names[j])
    ax[0].set_ylabel("boom angle (deg)" + ("\n(dashed = shaft command)" if MODE == "position" else ""))
    ax[1].set_ylabel("boom rate (rad/s)")
    ax[2].set_ylabel("hinge torque (N*m)")
    if MODE == "torque":
        ax[2].plot(t, L["cmd"], "--", lw=1, alpha=0.5, color="gray")
    finish(ax)

    # (2) bus attitude and rate
    fig, ax = new_fig("2_bus_attitude", 3, "Bus attitude and rate")
    for j, c in enumerate("wxyz"):
        ax[0].plot(t, L["quat"][:, j], label=f"q{c}")
    ax[0].set_ylabel("quaternion (wxyz)")
    for j, c in enumerate("xyz"):
        ax[1].plot(t, np.degrees(L["gyro"][:, j]), label=f"w{c}")
    ax[1].set_ylabel("gyro (deg/s)")
    ax[2].plot(t, quat_angle_deg(L["quat"], L["quat"][0]), label="angle from start")
    ax[2].set_ylabel("rotation from\ninitial attitude (deg)")
    finish(ax)

    # (3) motor internals
    n = 4 if MODE == "position" else 3
    fig, ax = new_fig("3_motor_internals", n, "Motor internals (output-referred torques)")
    for j in range(3):
        ax[0].plot(t, L["cog"][:, j], label=ax_names[j])
        ax[1].plot(t, L["drive"][:, j], label=ax_names[j])
    ax[0].set_ylabel("cogging torque (N*m)")
    ax[1].set_ylabel("electromagnetic\ndrive torque (N*m)")
    ax[2].plot(t, L["winding_w"], label="all 3 motors")
    ax[2].set_ylabel("winding loss (W)")
    if MODE == "position":
        for j in range(3):
            ax[3].plot(t, L["step_err_deg"][:, j], label=ax_names[j])
        ax[3].set_ylabel("field - rotor angle\n(electrical deg)")
        ax[3].axhline(90, color="r", lw=0.8, ls=":"); ax[3].axhline(-90, color="r", lw=0.8, ls=":")
    finish(ax)

    # (4) cogging close-up and spectra
    fs = 1.0 / (DT * LOG_EVERY)
    sel = t >= max(RAMP_S, 0.25 * T_END)
    fig = plt.figure(figsize=(10, 8.5)); figs["4_cogging_spectrum"] = fig
    a1 = fig.add_subplot(3, 1, 1)
    zoom = (t >= t[-1] - 0.05)
    for j in range(3):
        a1.plot((t[zoom] - t[zoom][0]) * 1e3, L["cog"][zoom, j], label=ax_names[j])
    a1.set_xlabel("ms from start of window"); a1.set_ylabel("cogging torque (N*m)")
    a1.set_title(f"Cogging, last 50 ms ({1e3 / fs:.1f} ms samples; use LOG_EVERY = 1 to resolve the waveform)")
    a2 = fig.add_subplot(3, 1, 2); a3 = fig.add_subplot(3, 1, 3, sharex=a2)
    if sel.sum() > 64:
        f, pc = psd(L["cog"][sel], fs)
        _, pg = psd(np.degrees(L["gyro"][sel]), fs)
        for j in range(3):
            a2.loglog(f[1:], pc[1:, j], label=ax_names[j])
            a3.loglog(f[1:], pg[1:, j], label=f"bus w{'xyz'[j]}")
        f_cog = MOTOR.cogging_freq_hz(np.max(np.abs(L["shaft_rate"][sel])))
        for a in (a2, a3):
            if 0 < f_cog < fs / 2:
                a.axvline(f_cog, color="r", lw=0.8, ls=":", label=f"cogging at peak gearbox-output speed ({f_cog:.0f} Hz)")
    a2.set_ylabel("cogging PSD\n(N*m)^2/Hz"); a3.set_ylabel("bus rate PSD\n(deg/s)^2/Hz")
    a3.set_xlabel("frequency (Hz)")
    for a in (a1, a2, a3):
        a.grid(alpha=0.3, which="both"); a.legend(loc="upper right", fontsize=8)

    if COMPLIANT:
        fig, ax = new_fig("6_coupling", 3, "Bus-to-motor compliant mounts")
        for j in range(3):
            ax[0].plot(t, np.degrees(L["coupling_deflection"][:, j]), label=ax_names[j])
            ax[1].plot(t, L["coupling_torque"][:, j], label=ax_names[j])
            ax[2].plot(t, L["coupling_loss_w"][:, j], label=ax_names[j])
        ax[0].set_ylabel("mount twist (deg)")
        ax[1].set_ylabel("torque on boom (N*m)")
        ax[2].set_ylabel("dissipated power (W)")
        finish(ax)

    # (5) orbit
    fig, ax = new_fig("5_orbit", 1, "Reference orbit", height=3.0)
    ax[0].plot(t, np.linalg.norm(L["R_eci"], axis=1) - R_EARTH, label="altitude (km)")
    ax[0].set_ylabel("chief altitude (km)")
    finish(ax)

    for name, fig in figs.items():
        fig.tight_layout()
        fig.savefig(save_dir / f"{name}.png", dpi=130)
        if not SHOW_PLOTS:
            plt.close(fig)
    print(f"Saved {len(figs)} figures to {save_dir}")
    if SHOW_PLOTS:
        plt.show()


def main():
    sim = BoomSimulation()
    model, data, motors = sim.model, sim.data, sim.motors
    print("central body:", model.central_body)
    # ---- 5. Sim loop ------------------------------------------------------------
    n_steps = int(round(T_END / DT))
    n_log = n_steps // LOG_EVERY + 1
    log = {name: np.full((n_log,) + shape, np.nan) for name, shape in {
        "t": (), "quat": (4,), "gyro": (3,), "R_eci": (3,),
        "boom_angle": (3,), "boom_rate": (3,),
        "shaft_angle": (3,), "shaft_rate": (3,), "coupling_deflection": (3,),
        "coupling_torque": (3,), "coupling_loss_w": (3,),
        "cmd": (3,), "tau": (3,), "cog": (3,), "drive": (3,),
        "winding_w": (), "step_err_deg": (3,),
    }.items()}
    n_row = 0

    for k in range(n_steps):
        cmd = sim.apply_control()
        should_log = k % LOG_EVERY == 0

        if should_log:
            theta = data.qpos[sim.shaft_q]
            omega = data.qvel[sim.shaft_v]
            deflection = data.qpos[sim.flex_q] if COMPLIANT else np.zeros(3)
            slip = data.qvel[sim.flex_v] if COMPLIANT else np.zeros(3)
            i = n_row
            log["t"][i] = data.time
            log["quat"][i] = data.qpos[3:7]
            log["gyro"][i] = data.sensors.measure("gyro", noisy=False)
            log["R_eci"][i] = data.orbit.R_eci
            log["shaft_angle"][i] = theta
            log["shaft_rate"][i] = omega
            log["coupling_deflection"][i] = deflection
            if COMPLIANT:
                log["coupling_torque"][i] = -sim.flex_stiffness * deflection - sim.flex_damping * slip
            log["coupling_loss_w"][i] = sim.flex_damping * slip**2 if COMPLIANT else 0.0
            log["boom_angle"][i] = theta + deflection
            log["boom_rate"][i] = omega + slip
            log["cmd"][i] = cmd
            if MODE == "position":              # field angle minus rotor angle, electrical degrees
                log["step_err_deg"][i] = np.degrees(
                    motors.phi - MOTOR.rotor_teeth * MOTOR.ratio * theta)

        mjo_step(model, data)

        if should_log:  # MuJoCo computes dcmotor torque and current during the step
            tau, drive = sim.motor_torques()
            log["tau"][i] = tau
            log["cog"][i] = sim.cogging_torque
            log["drive"][i] = drive
            log["winding_w"][i] = sim.winding_power()
            if not COMPLIANT:
                log["coupling_torque"][i] = tau
            n_row += 1

        if k % int(2.0 / DT) == 0:
            gyro = data.sensors.measure("gyro", noisy=False)
            print(f"t={data.time:5.1f}s  gyro={np.round(gyro, 4)}  "
                  f"quat={np.round(data.qpos[3:7], 3)}  "
                  f"booms(deg)={np.round(np.degrees(sim.boom_angles()), 1)}  "
                  f"cog={np.sqrt(np.mean(sim.cogging_torque ** 2)):.2f} N*m rms")

    print("Chief position (km):", data.orbit.R_eci)
    log = {k: v[:n_row] for k, v in log.items()}
    if SAVE_LOG:
        np.savez(SAVE_LOG, **log)
        print(f"Saved log to {SAVE_LOG} ({n_row} samples)")

    make_plots(log)

    # ---- 7. Sanity checks on the run --------------------------------------------
    w_peak = float(np.max(np.abs(log["shaft_rate"])))
    f_cog = MOTOR.cogging_freq_hz(w_peak)
    print(f"\n{MOTOR.name} peak output speed {w_peak:.3f} rad/s "
          f"(preset speed limit {MOTOR.max_output_speed:.2f} rad/s)")
    if COGGING and DETENT_MNM:
        spc = 1.0 / (f_cog * DT) if f_cog > 0 else np.inf
        print(f"cogging {f_cog:.0f} Hz at peak speed -> {spc:.1f} physics samples/cycle"
              + ("   WARNING: under-sampled, lower DT" if spc < 5 else ""))
    print(f"hinge torque: {np.sqrt(np.mean(log['tau'] ** 2)):.2f} N*m rms, {np.abs(log['tau']).max():.2f} N*m peak "
          f"(nominal output torque {MOTOR.nominal_output_torque:g} N*m)")
    print(f"cogging torque: {np.sqrt(np.mean(log['cog'] ** 2)):.2f} N*m rms")
    print(f"final body rate: {np.round(np.degrees(log['gyro'][-1]), 5)} deg/s")


if __name__ == "__main__":
    main()
