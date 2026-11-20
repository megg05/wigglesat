"""Two-phase geared stepper model with torque referred to the gearbox output.

StepperBank models, per motor, in the electrical-angle domain (theta_e = Nr * theta_motor):
  * two-phase torque production      T = Kt (-ia sin(theta_e) + ib cos(theta_e))
  * microstepped field angle, quantized to 90deg/microsteps, with a step-rate limit
  * current-loop lag and a supply-voltage limit on current amplitude (torque roll-off with speed)
  * cogging (detent) torque          T_cog = -T_d * sum_h a_h sin(h * theta_e + phase)
  * gearbox: T_out = efficiency * ratio * T_motor   (cogging is multiplied by the ratio too)
Lost steps / stalling / ringing after a step fall out of this naturally (the torque is periodic in
the rotor-to-field angle error). Two command modes:
  "position": open-loop stepper, command = output angle (rad)
  "torque"  : closed-loop commutation (field held 90 deg ahead of the rotor), command = output torque (N*m)

"""
from __future__ import annotations

import numpy as np

from .motor import GearedStepper


def configure_stepper_hinges(spec, motor: GearedStepper, names: tuple[str, ...]) -> None:
    """Set reflected inertia and the torque limit on named hinge actuators."""
    for name in names:
        actuator = spec.actuator(name)
        spec.joint(actuator.target).armature = motor.reflected_inertia
        limit = motor.emergency_output_torque
        actuator.ctrlrange = [-limit, limit]
        actuator.ctrllimited = 1


class StepperBank:
    """n identical geared steppers, vectorised. Call step() once per physics step."""

    def __init__(self, motor: GearedStepper, n: int = 3, *, mode: str = "position",
                 cogging: bool = True, detent_torque: float | None = None,
                 microsteps: int | None = None, phase_spread: float = 0.0, seed: int = 0):
        if mode not in ("position", "torque"):
            raise ValueError("mode must be 'position' or 'torque'")
        self.m, self.n, self.mode, self.cogging = motor, n, mode, cogging
        self.detent = motor.detent_torque if detent_torque is None else float(detent_torque)
        self.microsteps = int(microsteps or motor.microsteps)
        rng = np.random.default_rng(seed)
        self.psi = phase_spread * rng.uniform(-1.0, 1.0, n)    # per-motor detent phase error
        self.reset(np.zeros(n))

    def reset(self, theta_out) -> None:
        m = self.m
        q = np.pi / (2 * self.microsteps)
        th_e = m.rotor_teeth * m.ratio * np.asarray(theta_out, float)
        self.phi = np.round(th_e / q) * q          # field angle starts aligned with the rotor
        self.ia = np.zeros(self.n)
        self.ib = np.zeros(self.n)

    def cogging_torque_motor(self, th_e: np.ndarray) -> np.ndarray:
        if not self.cogging:
            return np.zeros_like(th_e)
        t = np.zeros_like(th_e)
        for order, rel, ph in self.m.detent_harmonics:
            t -= rel * np.sin(order * th_e + ph + self.psi)
        return self.detent * t

    def step(self, dt: float, theta_out, omega_out, cmd):
        """Advance the drive one physics step.

        theta_out, omega_out : gearbox output angle (rad) and rate (rad/s), arrays of length n
        cmd : output angle (rad) in 'position' mode, output torque (N*m) in 'torque' mode
        Returns (tau_out, cog_out, drive_out): torque applied at the gearbox output (N*m) and its
        cogging / electromagnetic parts, all output-referred.
        """
        m = self.m
        N, Nr, Kt, eta = m.ratio, m.rotor_teeth, m.torque_constant, m.efficiency
        th_e = Nr * N * np.asarray(theta_out, float)
        w_ms = N * np.asarray(omega_out, float)            # signed motor-shaft speed
        w_m = np.abs(w_ms)
        cmd = np.asarray(cmd, float)

        # supply-voltage limit on current amplitude (back-EMF + winding reactance roll-off)
        emf = Kt * w_m
        z = np.hypot(m.phase_resistance, Nr * w_m * m.inductance)
        i_lim = np.sqrt(np.maximum(m.supply_voltage ** 2 - emf ** 2, 0.0)) / z

        q = np.pi / (2 * self.microsteps)
        if self.mode == "position":
            target = Nr * N * cmd
            max_d = m.max_step_rate_hz * q * dt
            self.phi = self.phi + np.clip(target - self.phi, -max_d, max_d)
            phi_cmd = self.phi
            amp = np.full(self.n, m.run_current)
        else:
            tau_m = cmd / (eta * N)                       # motor-shaft torque wanted
            amp = np.abs(tau_m) / Kt
            phi_cmd = th_e + np.sign(tau_m) * 0.5 * np.pi  # field 90 deg ahead of the rotor
        amp = np.minimum(amp, i_lim)
        phi_q = np.round(phi_cmd / q) * q                 # microstep quantization

        alpha = 1.0 - np.exp(-dt * 2.0 * np.pi * m.current_loop_hz)
        self.ia += alpha * (amp * np.cos(phi_q) - self.ia)
        self.ib += alpha * (amp * np.sin(phi_q) - self.ib)

        t_drive = Kt * (-self.ia * np.sin(th_e) + self.ib * np.cos(th_e))
        t_cog = self.cogging_torque_motor(th_e)
        g = eta * N
        t_damp = -m.motor_damping * w_ms
        return g * (t_drive + t_cog + t_damp), g * t_cog, g * t_drive

    def winding_power(self) -> float:
        """Copper loss in W, summed over all motors and both phases."""
        return float(self.m.phase_resistance * np.sum(self.ia ** 2 + self.ib ** 2))
