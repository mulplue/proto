from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from isaac_utils import rotations, torch_utils

if TYPE_CHECKING:
    from phys_anim.envs.strike.isaacgym import StrikeHumanoid
else:
    StrikeHumanoid = object


class BaseStrike(StrikeHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_dist_min = 0.5
        self._tar_dist_max = 10.0
        self._near_dist = 1.5
        self._near_prob = 0.5
        
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        strike_body_names = self.config.strike_params.strike_body_names
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)

        self.strike_obs = torch.zeros(
            (self.config.num_envs, self.config.strike_params.obs_size),
            device=device,
            dtype=torch.float,
        )

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self.get_humanoid_root_states()[..., 0:3]
        return

    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)
        self.reset_target(env_ids)
        return

    def reset_target(self, env_ids):
        n = len(env_ids)

        init_near = torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) + self._tar_dist_min
        
        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + self.get_humanoid_root_states()[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + self.get_humanoid_root_states()[env_ids, 1]
        self._target_states[env_ids, 2] = 0.9
        
        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = rotations.quat_from_angle_axis(rand_rot_theta, axis, self.w_last)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0

    def compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self.get_humanoid_root_states()
            tar_states = self._target_states
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs = compute_strike_observations(root_states, tar_states, self.w_last)
        self.strike_obs[env_ids] = obs

    def compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        char_root_state = self.get_humanoid_root_states()
        strike_body_vel = self._rigid_body_vel[..., self._strike_body_ids[0], :]

        self.rew_buf[:] = compute_strike_reward(tar_pos, tar_rot, char_root_state, 
                                                self._prev_root_pos, strike_body_vel,
                                                self.dt, self._near_dist, self.w_last)

    def compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, self._tar_contact_forces,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights)
    


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_strike_observations(root_states, tar_states, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = rotations.quat_rotate(heading_rot, local_tar_pos, w_last)
    local_tar_vel = rotations.quat_rotate(heading_rot, tar_vel, w_last)
    local_tar_ang_vel = rotations.quat_rotate(heading_rot, tar_ang_vel, w_last)

    local_tar_rot = rotations.quat_mul(heading_rot, tar_rot, w_last)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot, w_last)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
    return obs

@torch.jit.script
def compute_strike_reward(tar_pos, tar_rot, root_state, prev_root_pos, strike_body_vel, dt, near_dist, w_last):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool) -> Tensor
    tar_speed = 1.0
    vel_err_scale = 4.0

    tar_rot_w = 0.6
    vel_reward_w = 0.4

    up = torch.zeros_like(tar_pos)
    up[..., -1] = 1
    tar_up = rotations.quat_rotate(tar_rot, up, w_last)
    tar_rot_err = torch.sum(up * tar_up, dim=-1)
    tar_rot_r = torch.clamp_min(1.0 - tar_rot_err, 0.0)

    root_pos = root_state[..., 0:3]
    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0


    reward = tar_rot_w * tar_rot_r + vel_reward_w * vel_reward
    
    succ = tar_rot_err < 0.2
    reward = torch.where(succ, torch.ones_like(reward), reward)

    return reward
    

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_contact_forces, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0
    
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        #strike_body_force = contact_buf[:, strike_body_id, :]
        #strike_body_has_contact = torch.any(torch.abs(strike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)
        
        has_failed = torch.logical_or(has_fallen, tar_fail)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated