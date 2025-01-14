from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from isaac_utils import rotations, torch_utils

if TYPE_CHECKING:
    from phys_anim.envs.heading.isaacgym import HeadingHumanoid
else:
    HeadingHumanoid = object


class BaseHeading(HeadingHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_speed_min = self.config.heading_params.tar_speed_min
        self._tar_speed_max = self.config.heading_params.tar_speed_max
        self._heading_change_steps_min = self.config.heading_params.heading_change_steps_min
        self._heading_change_steps_max = self.config.heading_params.heading_change_steps_max
        self._enable_rand_heading = self.config.heading_params.enable_rand_heading

        self.heading_obs = torch.zeros(
            (self.config.num_envs, self.config.heading_params.obs_size),
            device=device,
            dtype=torch.float,
        )

        self._heading_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self._tar_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_dir[..., 0] = 1.0
        
        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self.get_humanoid_root_states()[..., 0:3]
        return

    def reset_task(self, env_ids):
        super().reset_task(env_ids)

        n = len(env_ids)
        if n > 0:
            if (self._enable_rand_heading):
                rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
                rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            else:
                rand_theta = torch.zeros(n, device=self.device)
                rand_face_theta = torch.zeros(n, device=self.device)

            tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
            change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                        size=(n,), device=self.device, dtype=torch.int64)
            
            face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

            self._tar_speed[env_ids] = tar_speed
            self._tar_dir[env_ids] = tar_dir
            self._tar_facing_dir[env_ids] = face_tar_dir
            self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

    def update_task(self, actions):
        super().update_task(actions)
        
        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_task(rest_env_ids)
        return
    
    def compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self.get_humanoid_root_states()
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_facing_dir
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_facing_dir[env_ids]
        
        obs = compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir, self.w_last)
        self.heading_obs[env_ids] = obs

    def compute_reward(self, actions):
        root_pos = self.get_humanoid_root_states()[..., 0:3]
        root_rot = self.get_humanoid_root_states()[..., 3:7]
        self.rew_buf[:] = compute_heading_reward(root_pos, self._prev_root_pos,  root_rot,
                                                 self._tar_dir, self._tar_speed,
                                                 self._tar_facing_dir, self.dt, self.w_last)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir, w_last):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)
    
    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, w_last)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)
    
    tar_face_dir3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)
    local_tar_face_dir = rotations.quat_rotate(heading_rot, tar_face_dir3d, w_last)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]

    obs = torch.cat([local_tar_dir, tar_speed, local_tar_face_dir], dim=-1)
    return obs

@torch.jit.script
def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, tar_face_dir, dt, w_last):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool) -> Tensor
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    dir_reward_w = 0.7
    facing_reward_w = 0.3

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err + 
                        tangent_err_w * tangent_vel_err * tangent_vel_err))

    speed_mask = tar_dir_speed <= 0
    dir_reward[speed_mask] = 0

    heading_rot = torch_utils.calc_heading_quat(root_rot, w_last)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = rotations.quat_rotate(heading_rot, facing_dir, w_last)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward