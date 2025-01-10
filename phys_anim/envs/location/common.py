from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from isaac_utils import rotations, torch_utils

if TYPE_CHECKING:
    from phys_anim.envs.location.isaacgym import LocationHumanoid
else:
    LocationHumanoid = object
    

class BaseLocation(LocationHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_speed = self.config.location_params.tar_speed
        self._tar_change_steps_min = self.config.location_params.tar_change_steps_min
        self._tar_change_steps_max = self.config.location_params.tar_change_steps_max
        self._tar_dist_max = self.config.location_params.tar_dist_max

        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)

        self.location_obs = torch.zeros(
            (self.config.num_envs, self.config.location_params.obs_size),
            device=self.device,
            dtype=torch.float,
        )   # Jiahe: not sure

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self.get_humanoid_root_states()[..., 0:3]
        return
    
    ###############################################################
    # Handle resets
    ###############################################################

    def reset_task(self, env_ids):
        n = len(env_ids)

        char_root_pos = self.get_humanoid_root_states()[env_ids, 0:2]
        rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 2], device=self.device) - 1.0)

        change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        self._tar_pos[env_ids] = char_root_pos + rand_pos
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self.get_humanoid_root_states()
            tar_pos = self._tar_pos
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
            tar_pos = self._tar_pos[env_ids]
        
        obs = compute_location_observations(root_states, tar_pos, self.w_last)
        self.location_obs[env_ids] = obs
        return
    
    def compute_reward(self, actions):
        root_pos = self.get_humanoid_root_states()[..., 0:3]
        root_rot = self.get_humanoid_root_states()[..., 3:7]
        self.rew_buf[:] = compute_location_reward(root_pos, self._prev_root_pos, root_rot,
                                                 self._tar_pos, self._tar_speed,
                                                 self.dt, self.w_last)
        return

    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_states, tar_pos, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos3d = torch.cat([tar_pos, torch.zeros_like(tar_pos[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat(root_rot, w_last)
    
    local_tar_pos = rotations.quat_rotate(heading_rot, tar_pos3d - root_pos, w_last)
    local_tar_pos = local_tar_pos[..., 0:2]

    obs = local_tar_pos
    return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt, w_last):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, bool) -> Tensor
    dist_threshold = 0.5

    pos_err_scale = 0.5
    vel_err_scale = 4.0

    pos_reward_w = 0.5
    vel_reward_w = 0.4
    face_reward_w = 0.1
    
    pos_diff = tar_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0


    heading_rot = torch_utils.calc_heading_quat(root_rot, w_last)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = rotations.quat_rotate(heading_rot, facing_dir, w_last)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)


    dist_mask = pos_err < dist_threshold
    facing_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    return reward