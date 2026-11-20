"""Motor presets keyed by manufacturer part number.

Unspecified gearbox inertia is omitted. For integrated assemblies with no
mass breakdown, motor_mass stores the total assembly mass. Unmeasured detent
torque defaults to zero. Driver bandwidth and damping are simulation estimates.
"""
from .motor import GearedStepper


# https://www.linengineering.com/products/stepper-motors/hybrid-stepper-motors/5718-series/5718L-01P
# https://www.airengr.com/Lin-Engineering/Lin-Engineering-2011-2012-Catalog.pdf
LIN_5718L_HP50 = GearedStepper(
    name="Lin 5718L-01P-HP50",
    step_angle_deg=1.8,
    holding_torque=2.08,
    phase_current=2.8,
    phase_resistance=1.1,
    rotor_inertia=2.6 * 1.8290e-5,  # oz·in² to kg·m²
    motor_mass=1.07,
    ratio=50.0,
    efficiency=0.85,
    gearbox_inertia=0.03 * 1.0e-4,  # kg·cm² to kg·m²
    gearbox_mass=1.6,
    nominal_output_torque=50.0,
    emergency_output_torque=150.0,
    max_input_rpm=10_000.0,
    supply_voltage=24.0,  # Assumed drive supply.
    inductance=3.8e-3,  # Published for the parallel winding.
    current_loop_hz=1_000.0,
    microsteps=16,  # Assumed driver setting.
    max_step_rate_hz=100_000.0,  # Assumed driver limit.
    run_current=2.8,
    motor_damping=5.0e-3,  # Estimate.
    backlash_arcmin=7.0,
    detent_torque=0.04 * 2.08,  # Estimate.
)

# https://nmbtc.com/parts/pg15s-d20-hhb9/
# https://www.digikey.com/en/products/detail/nmb-technologies-corporation/PG15S-D20-HHB9/2417057
NMB_PG15S = GearedStepper(
    name="NMB PG15S-D20-HHB9",
    step_angle_deg=18.0,  # 20 motor steps/rev, before the 102.5:1 reduction.
    holding_torque=0.0105 / (102.5 * 0.70),  # Inferred from output holding torque.
    phase_current=0.100,
    phase_resistance=10.0,
    rotor_inertia=1.0e-8,  # Estimate.
    motor_mass=0.013,
    ratio=102.5,
    efficiency=0.70,  # Estimate.
    gearbox_inertia=0.0,
    gearbox_mass=0.0,
    nominal_output_torque=0.0105,  # Holding torque used as proxy for gearbox limit.
    emergency_output_torque=0.0105,
    max_input_rpm=47.0 * 102.5,
    supply_voltage=15.0,
    inductance=3.0e-3,  # Estimate.
    current_loop_hz=1_000.0,
    microsteps=1,  # Full-step simulation setting.
    max_step_rate_hz=1_600.0,
    run_current=0.100,
    motor_damping=1.0e-7,  # Estimate.
)

# https://file.globalso.com/file_manage/2297/20241022/kaifull-stepper-motor-catalog.pdf
KAIFULL_S60D120A = GearedStepper(
    name="Kaifull S60D120A-MAA00S2",
    step_angle_deg=1.8,
    holding_torque=4.0 / (100.0 * 0.75),  # Output allowable torque used as a holding proxy.
    phase_current=1.4,  # Full-winding configuration from the catalog.
    phase_resistance=2.2,
    rotor_inertia=135e-7,
    motor_mass=0.8,
    ratio=100.0,
    efficiency=0.75,  # Estimate.
    gearbox_inertia=0.0,
    gearbox_mass=0.0,
    nominal_output_torque=4.0,
    emergency_output_torque=4.0,
    max_input_rpm=18.0 * 100.0,  # Manufacturer's allowable output speed.
    supply_voltage=24.0,  # Catalog torque-speed curve test supply.
    inductance=4.4e-3,
    current_loop_hz=1_000.0,
    microsteps=16,  # Assumed driver setting.
    max_step_rate_hz=18.0 * 100.0 / 60.0 * 200 * 16,
    run_current=1.4,
    motor_damping=3.0e-5,  # Estimate.
)

# https://www.zaber.com/products/stepper-motors/X-NMS-E/specs?part=X-NMS17-E01
ZABER_X_NMS17 = GearedStepper(
    name="Zaber X-NMS17-E01-KX14C",
    step_angle_deg=360.0 / 200,
    holding_torque=0.58,  # Published maximum output torque used as a holding proxy.
    phase_current=2.3,
    phase_resistance=1.5,  # Effective winding estimate.
    rotor_inertia=8.0e-6,  # Estimate.
    motor_mass=0.58,
    ratio=1.0,
    efficiency=1.0,
    gearbox_inertia=0.0,
    gearbox_mass=0.0,
    nominal_output_torque=0.58,
    emergency_output_torque=0.58,
    max_input_rpm=3_000.0,
    supply_voltage=24.0,  # Lower end of the published supply range.
    inductance=3.0e-3,  # Effective winding estimate.
    current_loop_hz=1_000.0,
    microsteps=64,
    max_step_rate_hz=3_000.0 / 60.0 * 200 * 64,
    run_current=2.3,
    motor_damping=1.0e-4,  # Estimate.
)

# https://autonics.se/wp-content/uploads/2018/03/ak_r_rb_en_drw170419ab_20171207_he_20180109.pdf
# https://www.autonicschina.cn/cn/model/A8K-M566
# Equivalent two-phase model fitted to the five-phase unit's step spacing and
# torque scale. The simulated electrical angle is an equivalent field angle.
AUTONICS_A35K = GearedStepper(
    name="Autonics A35K-M566-R5",
    step_angle_deg=0.72,
    holding_torque=8.3 * 0.0980665,  # Related A8K-M566 bare motor, kgf·cm to N·m.
    phase_current=1.4,
    phase_resistance=1.1,
    rotor_inertia=280e-7,
    motor_mass=1.3,
    ratio=5.0,
    efficiency=(35.0 / 8.3) / 5.0,  # Inferred from output and bare-motor holding torque.
    gearbox_inertia=0.0,
    gearbox_mass=0.0,
    nominal_output_torque=35.0 * 0.0980665,
    emergency_output_torque=35.0 * 0.0980665,
    max_input_rpm=360.0 * 5.0,
    backlash_arcmin=20.0,
    supply_voltage=24.0,  # Assumed drive supply.
    inductance=3.0e-3,  # Effective winding estimate.
    current_loop_hz=1_000.0,
    microsteps=1,  # Full-step simulation setting.
    max_step_rate_hz=360.0 / 60.0 * 2_500,
    run_current=1.4,
    motor_damping=2.0e-4,  # Estimate.
)

MOTOR_PRESETS = {
    "5718L-01P-HP50": LIN_5718L_HP50,
    "PG15S-D20-HHB9": NMB_PG15S,
    "S60D120A-MAA00S2": KAIFULL_S60D120A,
    "X-NMS17-E01-KX14C": ZABER_X_NMS17,
    "A35K-M566-R5": AUTONICS_A35K,
}
