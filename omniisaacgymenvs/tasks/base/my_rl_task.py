import torch

from gym import spaces

from omni.isaac.gym.tasks.rl_task import RLTaskInterface
from omni.isaac.core.tasks import BaseTask
from omniisaacgymenvs.utils.domain_randomization.randomization import Randomizer


class RLTask(RLTaskInterface):

    """
    This class provides a PyTorch RL-specific interface for setting up RL tasks.
    It includes utilities for setting up RL task related parameters, clonning environments,
    and data collection for RL algorithms updates.
    """

    def __init__(self, name, env, offset=None) -> None:

        BaseTask.__init__(self, name=name, offset=offset)

        self._rand_seed = self._cfg["seed"]
        torch._C._jit_set_nvfuser_enabled(False)

        self.test = self._cfg["test"]
        self._device = self._cfg["sim_device"]

        # setup randomizer for domain randomization
        self._dr_randomizer = Randomizer(self._cfg, self._task_cfg)
        if self._dr_randomizer.randomize:
            import  omni.replicator.isaac as dr
            self.dr = dr

        # setup replicator for camera data collection
        if self._task_cfg["sim"].get("enable_cameras", False):
            from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
            from omni.replicator.isaac.scripts.writers.pytroch_listener import PytorchListnener
            import omni.replicator.core as rep

            self.rep = rep
            self.PytorchWriter = PytorchWriter
            self.PytorchListnener = PytorchListnener

        print("Task Device", self_device)

        self.randomize_actions = False
        self.randomize_observations = False

        self.clip_obs = self._task_cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self._task_cfg["env"].get("clipActions", np.Inf)
        self.rl_device = self._cfg.get("rl_device", "cuda:0")

        self.control_frequency_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        self.rendering_interval = self._task_cfg.get("renderingInterval", 1)

        print("RL device: ", self.rl_device)

        self._env = env

        if 

