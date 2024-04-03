import datetime
import os

import hydra
from omegaconf import DictConfig
import torch

from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path, get_experience
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUEnv

from rl_games.common import vecenv, env_configurations
from rl_games.torch_runner import Runner





class RLGTrainer:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict
        return
    
    def launch_rlg_hydra(self, env):
        self.cfg_dict["task"]["test"] = self.cfg.test
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: env})

        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)
        return
    
    def run(self):
        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())







@hydra.main(version_base=None, config_name="config",config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # local rank (GPU id) in a current multi-gpu mode
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # global rank (GPU id) in multi-gpu multi-node mode
    global_rank = int(os.getenv("RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = local_rank
        cfg.rl_device = f'cuda:{local_rank}'

    # select kit app file
    experience = get_experience(headless=cfg.headless, 
                                enable_livestream=cfg.enable_livestream, 
                                enable_viewport=cfg.task.sim.enable_cameras, 
                                kit_app=cfg.kit_app)

    env = VecEnvRLGames(
        headless=cfg.headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=cfg.task.sim.enable_cameras,
        experience=experience
    )

    # ensure checkpoints can can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoints is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    

    # set seed. if seed is -1 => will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    if cfg.seed != -1:
        cfg.seed += global_rank
    
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.setdefault

    task = initialize_task(cfg_dict, env)

    torch.cuda.set_device(local_rank)
    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg.trainer.run

    # [TODO] continue here!

    return







if __name__== "__main__":
    parse_hydra_configs()