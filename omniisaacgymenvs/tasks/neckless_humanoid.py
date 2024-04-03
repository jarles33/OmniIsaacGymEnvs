
import torch

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView

from omniisaacgymenvs.tasks.shared.locomotion import LocomotionTask
from omniisaacgymenvs.robots.articulations.neckless_humanoid import NecklessHumanoid
from omniisaacgymenvs.tasks.base.rl_task import RLTask


class NecklessHumanoidLocomotionTask(LocomotionTask):

    # __init__()
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._num_observations = 60
        self._num_actions = 12
        self._humanoid_positions = torch.tensor([0, 0, 0.68])
        self._humanoid_orientation = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])       # align to x direction
        # self._humanoid_orientation = torch.tensor([0.6087614, 0.0, 0.0, 0.7933533])

        LocomotionTask.__init__(self, name=name, env=env)
        return
    
    def update_config(self, sim_config):
        self._sim_config = sim_config               # this is of class SimConfig
        self._cfg = sim_config.config               # this is of class python Dict
        self._task_cfg = sim_config.task_config     # this is of class python Dict
        LocomotionTask.update_config(self)


    # set_up_scene()
    def set_up_scene(self, scene) -> None:
        self.get_neckless_humanoid()
        RLTask.set_up_scene(self, scene)
        self._neckless_humanoids = ArticulationView(
            prim_paths_expr="/World/envs/.*/NecklessHumanoid/body_upper",
            name="neckless_humanoid_view",
            reset_xform_properties=False
        )
        scene.add(self._neckless_humanoids)
        return


    def get_neckless_humanoid(self):
        neckless_humanoid = NecklessHumanoid(
            prim_path=self.default_zero_env_path + "/NecklessHumanoid",
            name="NecklessHumanoid",
            translation=self._humanoid_positions,
            orientation=self._humanoid_orientation,
        )

        self._sim_config.apply_articulation_settings(
            name="NecklessHumanoid", 
            prim=get_prim_at_path(neckless_humanoid.prim_path), 
            cfg=self._sim_config.parse_actor_config("NecklessHumanoid")
        )
        return
    
    def get_robot(self):
        return self._neckless_humanoids

    # initialize_views()
    def initialize_views(self, scene):
        RLTask.initialize_views(self, scene)
        if scene.object_exists("neckless_humanoid_view"):
            scene.remove_object("neckless_humanoid_view", registry_only=True)
        self._neckless_humanoids = ArticulationView(
            prim_paths_expr="/World/envs/.*/NecklessHumanoid/body_upper",
            name="neckless_humanoid_view",
            reset_xform_properties=False
        )
        scene.add(self._neckless_humanoids)

        return
    
    def post_reset(self):
        self.joint_gears = torch.tensor(
            [
                100.0,  # joint1
                100.0,  # joint2
                100.0,  # joint3
                100.0,  # joint4
                100.0,  # joint5
                100.0,  # joint6
                100.0,  # joint7
                100.0,  # joint8
                100.0,  # joint9
                100.0,  # joint10
                100.0,  # joint11
                100.0,  # joint12
            ],
            device=self._device,
        )
        self.max_motor_effort = torch.max(self.joint_gears)
        self.motor_effort_ratio = self.joint_gears / self.max_motor_effort
        dof_limits = self._neckless_humanoids.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        force_links = ["foot_l", "foot_r"]
        self._sensor_indices = torch.tensor(
            [self._neckless_humanoids._body_indices[j] for j in force_links], device=self._device, dtype=torch.long
        )

        LocomotionTask.post_reset(self)
        # address the face neckless humanoid does not face X direction
        self.heading_vec = torch.tensor([0, -1, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        return
    
    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(obs_buf=self.obs_buf,
                                     motor_effort_ratio=self.motor_effort_ratio,
                                     joints_at_limit_cost_scale=self.joints_at_limit_cost_scale)
    
    def calculate_metrics(self) -> None:
        super().calculate_metrics()

        symmetry_cost_scale = 0.01
        symmetry_cost = symmetry_cost_scale * (self.obs_buf[:,0] + self.obs_buf[:,1]).abs()                     # encourage shoulds to move symmetrically
        symmetry_cost += symmetry_cost_scale * (self.obs_buf[:,4] + self.obs_buf[:,5]).abs()                    # encourage legs to move symmetrically

        # get a sense how much impact this newly introduced symmetry_cost has on total metrics
        print(f"ratio of symmetry_cost/total_cost = {(symmetry_cost / self.rew_buf).mean() * 100.0}")


    

@torch.jit.script
def get_dof_at_limit_cost(obs_buf, motor_effort_ratio, joints_at_limit_cost_scale):
    # type: (Tensor, Tensor, float) -> Tensor
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:(12+12)]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum(
        (torch.abs(obs_buf[:, 12:(12+12)]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1
    )
    return dof_at_limit_cost