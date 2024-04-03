from typing import Optional

import carb
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage


class NecklessHumanoid(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "NecklessHumanoid",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = "omniverse://localhost/Users/jarles/Robot Models/neckless_humanoid_v2/robot/robot.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        # add environment
        # add_reference_to_stage("omniverse://localhost/Users/jarles/Environments/Track/Track.usd", prim_path+"_env")

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )