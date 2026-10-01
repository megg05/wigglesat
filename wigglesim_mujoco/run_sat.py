"""Starter script: open-loop boom motion, check that angular momentum is conserved.

    pip install mujoco numpy
    python run_sat.py            # headless, prints a summary
    python run_sat.py --viewer   # interactive viewer
"""
import sys
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("sat_booms.xml")
data = mujoco.MjData(model)

bus_id = model.body("bus").id


def sensor(name):
    s = model.sensor(name)
    return data.sensordata[s.adr[0]: s.adr[0] + s.dim[0]]


def system_angmom():
    """Total angular momentum about the system COM (world frame)."""
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_angmom[bus_id].copy()


def controller(m, d):
    """Placeholder open-loop command: phase-shifted sinusoids on each boom (N*m)."""
    t = d.time
    d.ctrl[0] = 0.2 * np.sin(2 * np.pi * 0.5 * t)
    d.ctrl[1] = 0.2 * np.sin(2 * np.pi * 0.5 * t + 2.0)
    d.ctrl[2] = 0.2 * np.sin(2 * np.pi * 0.5 * t + 4.0)


mujoco.set_mjcb_control(controller)

if "--viewer" in sys.argv:
    import mujoco.viewer
    mujoco.viewer.launch(model, data)
else:
    mujoco.mj_forward(model, data)
    L0 = system_angmom()
    for i in range(int(10.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if i % 1000 == 0:
            print(f"t={data.time:5.2f}  gyro={np.round(sensor('gyro'), 4)}  "
                  f"quat={np.round(sensor('quat'), 3)}")
    print("Initial L:", L0)
    print("Final   L:", system_angmom(), "(should be ~constant: no external torques)")