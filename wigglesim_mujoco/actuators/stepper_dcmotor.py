"""Geared stepper with open-loop commutation and MuJoCo winding dynamics.

Call configure_dcmotors on the MuJoCo spec before compiling,
then create a DCStepperBank from the compiled model. The bank writes voltages
to ctrl. Read actuator_force times the gear ratio for gearbox-output torque.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .motor import GearedStepper


def configure_dcmotors(
    spec: mujoco.MjSpec,
    motor: GearedStepper,
    names: tuple[str, ...],
    *,
    cogging: bool = False,
) -> None:
    """Convert named hinge actuators to dcmotors with optional dominant-harmonic cogging."""
    kt = motor.torque_constant
    detent = [0.0, 0.0, 0.0]
    if cogging and motor.detent_torque and motor.detent_harmonics:
        order, amplitude, phase = max(motor.detent_harmonics, key=lambda harmonic: abs(harmonic[1]))
        detent = [
            motor.efficiency * motor.detent_torque * abs(amplitude),
            order * motor.rotor_teeth,
            phase + (np.pi if amplitude >= 0 else 0),
        ]
    for name in names:
        actuator = spec.actuator(name)
        joint = spec.joint(actuator.target)
        # Actuator position and velocity are motor-side. Apply gearbox efficiency
        # to the torque constant and detent amplitude, while using Kt as a Ke estimate.
        # MuJoCo's cogging term is +A sin(Np * length + phase).
        actuator.set_to_dcmotor(
            motorconst=[motor.efficiency * kt, kt],
            resistance=motor.phase_resistance,
            inductance=[motor.inductance, 0.0],
            cogging=detent,
            saturation=[motor.emergency_output_torque / motor.ratio, 0.0, 0.0],
        )
        actuator.gear = [motor.ratio, 0, 0, 0, 0, 0]
        actuator.ctrlrange = [-motor.supply_voltage, motor.supply_voltage]
        actuator.ctrllimited = 1
        joint.armature = motor.reflected_inertia
        joint.damping[0] += motor.efficiency * motor.ratio**2 * motor.motor_damping


class DCStepperBank:
    """Drive configured dcmotors with one effective torque-producing current each.

    Call step before each physics step. A position command is an unwrapped
    gearbox-output angle in rad. A torque command is in gearbox-output N*m.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        motor: GearedStepper,
        names: tuple[str, ...],
        *,
        mode: str = "position",
        microsteps: int | None = None,
    ):
        if mode not in ("position", "torque"):
            raise ValueError("mode must be 'position' or 'torque'")
        self.motor = motor
        self.mode = mode
        self.microsteps = motor.microsteps if microsteps is None else microsteps
        if self.microsteps < 1:
            raise ValueError("microsteps must be positive")
        self._quantum = np.pi / (2 * self.microsteps)
        self._dt = model.opt.timestep
        self._current_response = -np.expm1(-self._dt * motor.phase_resistance / motor.inductance)
        self._electrical_ratio = motor.rotor_teeth * motor.ratio
        self.actuators = np.array([model.actuator(name).id for name in names], dtype=int)
        joints = model.actuator_trnid[self.actuators, 0]
        self._qpos = model.jnt_qposadr[joints]
        self._qvel = model.jnt_dofadr[joints]
        self._current = model.actuator_actadr[self.actuators]
        if np.any(model.actuator_dyntype[self.actuators] != mujoco.mjtDyn.mjDYN_DCMOTOR):
            raise ValueError("configure the dcmotor actuators before compiling the model")
        self.phi = None
        # The first three dcmotor bias parameters are cogging amplitude, order, and phase.
        self._cog_amplitude, self._cog_order, self._cog_phase = (
            model.actuator_biasprm[self.actuators, :3].T
        )

    def cogging_torque(self, data: mujoco.MjData) -> np.ndarray:
        """Return gearbox-output cogging torque at the current joint positions."""
        angle_motor = self.motor.ratio * data.qpos[self._qpos]
        return self.motor.ratio * self._cog_amplitude * np.sin(
            self._cog_order * angle_motor + self._cog_phase
        )

    def winding_power(self, data: mujoco.MjData) -> float:
        """Return copper loss for the effective current states."""
        return float(self.motor.phase_resistance * np.sum(data.act[self._current] ** 2))

    def reset(self, data: mujoco.MjData) -> None:
        """Align the field and clear the effective winding current."""
        theta_e = self._electrical_ratio * data.qpos[self._qpos]
        self.phi = np.rint(theta_e / self._quantum) * self._quantum
        self._field_progress = self.phi.copy()
        data.act[self._current] = 0.0
        data.ctrl[self.actuators] = 0.0

    def step(self, data: mujoco.MjData, cmd) -> None:
        """Write terminal voltages to ctrl. MuJoCo integrates current and applies torque."""
        if self.phi is None:
            self.reset(data)
        m = self.motor
        cmd = np.broadcast_to(np.asarray(cmd, dtype=float), self.actuators.shape)
        omega_motor = m.ratio * data.qvel[self._qvel]
        if self.mode == "position":
            target = self._electrical_ratio * cmd
            max_advance = m.max_step_rate_hz * self._quantum * self._dt
            self._field_progress += np.clip(
                target - self._field_progress, -max_advance, max_advance
            )
            self.phi = np.rint(self._field_progress / self._quantum) * self._quantum
            theta_e = self._electrical_ratio * data.qpos[self._qpos]
            current = m.run_current * np.sin(self.phi - theta_e)
        else:
            current = cmd / (m.efficiency * m.ratio * m.torque_constant)
            current = np.clip(current, -m.run_current, m.run_current)

        # Reserve voltage for rotating the phase-current vector at speed. The
        # scalar winding state alone captures lag along the torque axis.
        emf = m.torque_constant * omega_motor
        impedance = np.hypot(m.phase_resistance, m.rotor_teeth * omega_motor * m.inductance)
        headroom = np.sqrt(np.maximum(m.supply_voltage**2 - emf**2, 0.0))
        current = np.clip(current, -headroom / impedance, headroom / impedance)
        # A current-regulating driver uses the winding state to choose voltage.
        # Invert the one-step RL response, leaving the supply clamp to limit slew.
        measured = data.act[self._current]
        voltage = m.phase_resistance * (measured + (current - measured) / self._current_response) + emf
        data.ctrl[self.actuators] = np.clip(voltage, -m.supply_voltage, m.supply_voltage)
