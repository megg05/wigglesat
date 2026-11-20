"""Parameters shared by the two geared stepper models."""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd, pi, sqrt


@dataclass(frozen=True)
class GearedStepper:
    """Two-phase motor, gearbox, and driver settings in SI unless noted."""

    name: str
    step_angle_deg: float
    holding_torque: float  # both phases energized at phase_current
    phase_current: float
    phase_resistance: float
    rotor_inertia: float
    motor_mass: float

    ratio: float
    efficiency: float
    gearbox_inertia: float
    gearbox_mass: float
    nominal_output_torque: float
    emergency_output_torque: float
    max_input_rpm: float

    supply_voltage: float
    inductance: float
    current_loop_hz: float
    microsteps: int
    max_step_rate_hz: float  # microsteps/second
    run_current: float
    motor_damping: float  # at motor shaft

    backlash_arcmin: float = 0.0
    detent_torque: float = 0.0  # at motor shaft
    detent_harmonics: tuple[tuple[int, float, float], ...] = ((4, 1.0, 0.0),)    # (electrical harmonic order, relative amplitude, phase in rad)


    @property
    def rotor_teeth(self) -> int:
        return int(round(90.0 / self.step_angle_deg))

    @property
    def torque_constant(self) -> float:
        """N*m/A per phase, from holding torque = sqrt(2) * Kt * I (both phases on)."""
        return self.holding_torque / (sqrt(2.0) * self.phase_current)

    @property
    def reflected_inertia(self) -> float:
        """Rotor + gearbox input inertia seen at the output shaft, kg*m^2 (use as joint armature)."""
        return (self.rotor_inertia + self.gearbox_inertia) * self.ratio ** 2

    @property
    def total_mass(self) -> float:
        return self.motor_mass + self.gearbox_mass

    @property
    def max_output_speed(self) -> float:
        return self.max_input_rpm / self.ratio * 2.0 * pi / 60.0

    @property
    def full_step_output_deg(self) -> float:
        return self.step_angle_deg / self.ratio

    @property
    def cogging_output_amplitude(self) -> float:
        """Peak cogging torque at the gearbox output, N*m."""
        return self.detent_torque * self.ratio * self.efficiency

    def cogging_freq_hz(self, omega_out: float) -> float:
        """Fundamental cogging frequency (Hz) at output speed omega_out (rad/s)."""
        order = gcd(*(order for order, amplitude, _ in self.detent_harmonics if amplitude))
        return abs(omega_out) * order * self.rotor_teeth * self.ratio / (2.0 * pi)

