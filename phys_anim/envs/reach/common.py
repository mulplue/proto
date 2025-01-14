from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from isaac_utils import rotations, torch_utils

if TYPE_CHECKING:
    from phys_anim.envs.reach.isaacgym import ReachHumanoid
else:
    ReachHumanoid = object


class BaseReach(ReachHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_speed = self.config.reach_params.tar_speed
        self._tar_change_steps_min = self.config.reach_params.tar_change_steps_min
        self._tar_change_steps_max = self.config.reach_params.tar_change_steps_max
        self._tar_dist_max = self.config.reach_params.tar_dist_max
        self._tar_height_min = self.config.reach_params.tar_height_min
        self._tar_height_max = self.config.reach_params.tar_height_max

        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        reach_body_name = self.config.reach_params.reach_body_name
        self._reach_body_id = self._build_reach_body_id_tensor(self.envs[0], self.humanoid_handles[0], reach_body_name)

        self.reach_obs = torch.zeros(
            (self.config.num_envs, self.config.reach_params.obs_size),
            device=device,
            dtype=torch.float,
        )

    def reset_task(self, env_ids):
        super().reset_task(env_ids)

        n = len(env_ids)
        if n > 0:
            rand_pos = torch.rand([n, 3], device=self.device)
            rand_pos[..., 0:2] = self._tar_dist_max * (2.0 * rand_pos[..., 0:2] - 1.0)
            rand_pos[..., 2] = (self._tar_height_max - self._tar_height_min) * rand_pos[..., 2] + self._tar_height_min
            
            change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                        size=(n,), device=self.device, dtype=torch.int64)

            self._tar_pos[env_ids, :] = rand_pos
            self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        
    
    def update_task(self, actions):
        super().update_task(actions)

        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_task(rest_env_ids)

    def compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self.get_humanoid_root_states()
            tar_pos = self._tar_pos
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
            tar_pos = self._tar_pos[env_ids]
        
        obs = compute_location_observations(root_states, tar_pos, self.w_last)
        self.reach_obs[env_ids] = obs

    def compute_reward(self, actions):
        reach_body_pos = self.rigid_body_pos[:, self._reach_body_id, :]
        root_rot = self.get_humanoid_root_states()[..., 3:7]
        self.rew_buf[:] = compute_reach_reward(reach_body_pos, root_rot,
                                                 self._tar_pos, self._tar_speed,
                                                 self.dt)
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_states, tar_pos, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)
    local_tar_pos = rotations.quat_rotate(heading_rot, tar_pos, w_last)

    obs = local_tar_pos
    return obs

# @torch.jit.script
def compute_reach_reward(reach_body_pos, root_rot, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    pos_err_scale = 4.0
    
    pos_diff = tar_pos - reach_body_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    
    reward = pos_reward

    return reward