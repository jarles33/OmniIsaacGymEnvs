import torch

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole


class CartpoleTask(RLTask):
    ##############
    ## __init__ ##
    ##############
    def __init__(self, name, sim_config, env, offset_None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        RLTask.__init__(self, name, env)
        return
    
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    ##################
    ## set_up_scene ##
    ##################
    def set_up_scene(self, scene) -> None:
        self.get_cartpole()



    def get_cartpole(self):
        cartpole = Cartpole(prim_path=self.default_zero_env_path+"/Cartpole",
                            name="Cartpole",
                            translation=self._cartpole_positions)
        
        # apply articulation 
        self._sim_config.apply_articulation_settings()