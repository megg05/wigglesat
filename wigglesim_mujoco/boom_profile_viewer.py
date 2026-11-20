"""python -m wigglesim_mujoco.boom_profile_viewer"""
from __future__ import annotations

import sys
from importlib.metadata import entry_points

import numpy as np
from mjorbit import MjoData, MjoModel
from mjorbit.constants import R_EARTH
from viewer.tasks import ViewerTask, register_task

from wigglesim_mujoco.boom_profile import BoomSimulation


@register_task
class BoomProfileTask(ViewerTask):
    name = "boom_profile"
    description = "Stepper-driven booms using the settings in boom_profile.py."
    track_body = "sc_bus"

    def build(self) -> tuple[MjoModel, MjoData]:
        self.sim = BoomSimulation()
        return self.sim.model, self.sim.data

    def pre_step(self) -> None:
        self.sim.apply_control()

    def status(self) -> str:
        body_rate = np.degrees(self.data.qvel[3:6])
        booms = np.degrees(self.sim.boom_angles())
        alt = np.linalg.norm(self.data.orbit.R_eci) - R_EARTH
        cogging_rms = np.sqrt(np.mean(self.sim.cogging_torque ** 2))
        return (
            f"**motor** {self.sim.motor.name} ({self.sim.backend}, {self.sim.motors.mode} mode)<br>"
            f"**body rate** = [{body_rate[0]:.3f}, {body_rate[1]:.3f}, {body_rate[2]:.3f}] deg/s &nbsp; "
            f"**booms** = [{booms[0]:.1f}, {booms[1]:.1f}, {booms[2]:.1f}] deg &nbsp; "
            f"**altitude** = {alt:.0f} km<br>"
            f"**cogging** {cogging_rms:.2f} N*m rms &nbsp; "
            f"**winding loss** = {self.sim.winding_power():.1f} W"
        )


def viewer_main():
    """The callable behind the installed `mjo-viewer` command."""
    for ep in entry_points(group="console_scripts"):
        if ep.name == "mjo-viewer":
            return ep.load()
    raise SystemExit("Could not find the 'mjo-viewer' entry point; is mjorbit installed here?")


if __name__ == "__main__":
    if "--task" not in sys.argv and "--list-tasks" not in sys.argv:
        sys.argv += ["--task", BoomProfileTask.name]
    sys.exit(viewer_main()())
