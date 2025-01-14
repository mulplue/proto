from isaac_utils import torch_utils, rotations

import json
import torch
from torch import Tensor
import numpy as np

from phys_anim.envs.env_utils.path_generator import PathGenerator
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from phys_anim.envs.unihsi.isaacgym import UnihsiHumanoid
else:
    UnihsiHumanoid = object


class BaseUnihsi(UnihsiHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        
        
        strike_body_names = self.config.unihsi_params.strike_body_names

        self._tar_dist_min = 0.5
        self._tar_dist_max = 10.0
        self._near_dist = 1.5
        self._near_prob = 0.5

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)

        self.contact_type = torch.zeros([self.num_envs, self.joint_num], device=self.device, dtype=torch.bool)
        self.contact_valid = torch.zeros([self.num_envs, self.joint_num], device=self.device, dtype=torch.bool)

        self.joint_diff_buff = torch.ones([self.num_envs, self.joint_num], device=self.device, dtype=torch.float)
        self.location_diff_buf = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self.joint_idx_buff = torch.ones([self.num_envs, self.joint_num], device=self.device, dtype=torch.long)
        self.tar_dir = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
    
        self.envs_idx = torch.arange(self.num_envs).to(self.device)

        self.pelvis2torso = torch.tensor([0,0,0.236151]).to(self.device)[None].repeat(self.num_envs, 1)
        self.pelvis2torso[:, 2] += 0.15
        self.torso2head = torch.tensor([0, 0, 0.223894]).to(self.device)[None].repeat(self.num_envs, 1)
        self.torso2head[:, 2] += 0.15
        self.new_rigid_body_pos = torch.ones([self.num_envs, 15, 3], device=self.device, dtype=torch.float)

        self.step_mode = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self.change_obj = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.stand_point_choice = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)

        self.big_force = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        
        self.still = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.still_buf = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)


        self.unihsi_obs = torch.zeros([self.num_envs, self.config.unihsi_params.obs_size], device=self.device, dtype=torch.float)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self.get_humanoid_root_states()[..., 0:3]

    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)
        if len(env_ids) > 0:
            success = (self.location_diff_buf[env_ids] < 0.1) & ~self.big_force[env_ids]

        self.reset_target(env_ids, success)

    def reset_target(self, env_ids, success):

        contact_type_steps = self.contact_type_step[self.scene_for_env, self.step_mode]
        contact_valid_steps = self.contact_valid_step[self.scene_for_env, self.step_mode]
        fulfill = ((contact_valid_steps & \
                     (((contact_type_steps) & (self.joint_diff_buff < 0.1)) | (((~contact_type_steps) & (self.joint_diff_buff >= 0.05))))) \
                        | (~contact_valid_steps))[env_ids] & (success[:, None]) # need add contact direction
        fulfill = torch.all(fulfill, dim=-1)

        self.step_mode[env_ids[fulfill]] += 1

        max_step = self.step_mode[env_ids] == self.max_steps[self.scene_for_env][env_ids]


        reset = ~fulfill | max_step
        super().reset_actors(env_ids[reset])

        self.still_buf[env_ids[reset|fulfill]] = 0

        rand_rot_theta = 2 * np.pi * torch.rand([self.num_envs], device=self.device)
        axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        rand_rot = rotations.quat_from_angle_axis(rand_rot_theta, axis, self.w_last)
        self.get_humanoid_root_states()[env_ids[reset], 3:7] = rand_rot[env_ids[reset]]

        dist_max = 4
        dist_min = 2
        rand_dist_y = (dist_max - dist_min) * torch.rand([self.num_envs], device=self.device) + dist_min
        rand_dist_x = (dist_max - dist_min) * torch.rand([self.num_envs], device=self.device) + dist_min
        x_sign = torch.from_numpy(np.random.choice((-1, 1), [self.num_envs])).to(self.device)
        y_sign = torch.from_numpy(np.random.choice((-1, 1), [self.num_envs])).to(self.device)
        self.get_humanoid_root_states()[env_ids[reset], 0] += self.x_offset[env_ids[reset]] + x_sign[env_ids[reset]] * rand_dist_x[env_ids[reset]]
        self.get_humanoid_root_states()[env_ids[reset], 1] += self.y_offset[env_ids[reset]] + y_sign[env_ids[reset]] * rand_dist_y[env_ids[reset]] - 2
        self.step_mode[env_ids[reset]] = 0

        stand_point_choice = torch.from_numpy(np.random.choice((0,1,2,3), [self.num_envs])).to(self.device)
        self.stand_point_choice[env_ids[reset]] = stand_point_choice[env_ids[reset]]

        self.contact_type = self.contact_type_step[self.scene_for_env, self.step_mode]
        self.contact_valid = self.contact_valid_step[self.scene_for_env, self.step_mode]
        self.contact_direction = self.contact_direction_step[self.scene_for_env, self.step_mode]

        self.stand_point = self.scene_stand_point[range(len(self.step_mode)), self.step_mode, self.stand_point_choice]

        self.envs_obj_pcd_buffer[env_ids] = self.obj_pcd_buffer[self.scene_for_env[env_ids], self.step_mode[env_ids]]
        self.envs_obj_pcd_buffer[env_ids] = torch.einsum("nmoe,neg->nmog", self.envs_obj_pcd_buffer[env_ids], self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col][env_ids])
        self.envs_obj_pcd_buffer[env_ids, ..., 0] += self.x_offset[:, None, None][env_ids] + self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]
        self.envs_obj_pcd_buffer[env_ids, ..., 1] += self.y_offset[:, None, None][env_ids] + self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]
        self.envs_obj_pcd_buffer[env_ids, ..., 2] += self.rand_dist_z[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]


    def compute_task_obs(self, env_ids=None):

        pcd_buffer = []
        self.new_rigid_body_pos = self._rigid_body_pos.clone()

        head = rotations.quat_rotate(self._rigid_body_rot[:, 1], self.torso2head, self.w_last)
        head += self._rigid_body_pos[:, 1]
        self.new_rigid_body_pos[:, 2] = head

        torso = rotations.quat_rotate(self._rigid_body_rot[:, 0], self.pelvis2torso, self.w_last)
        torso += self._rigid_body_pos[:, 0]
        self.new_rigid_body_pos[:, 1] = torso
        joint_pos_buffer = []
        if (env_ids is None):
            root_states = self.get_humanoid_root_states()
            tar_pos = self.stand_point
            pcd_buffer = self.envs_obj_pcd_buffer[self.envs_idx]
            env_num, joint_num, point_num, point_dim = pcd_buffer.shape
            pcd_buffer = pcd_buffer.view(-1, point_num, point_dim)
            pcd_buffer = pcd_buffer[range(pcd_buffer.shape[0]), self.joint_idx_buff.view(-1)]
            pcd_buffer = pcd_buffer.reshape(env_num, joint_num, point_dim)
            joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids]

            joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode]
            valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode]
            joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
            pcd_buffer_view = pcd_buffer.view(-1, 3)
            pcd_buffer_view[valid_joint_contact_choice.view(-1)] = joints_contact[valid_joint_contact_choice.view(-1)]

            height_map = self.envs_heightmap

            contact_type = self.contact_type
            contact_valid = self.contact_valid
            contact_direction = self.contact_direction

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset
            tar_dir = self.tar_dir
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
            tar_pos = self.stand_point[env_ids]
            pcd_buffer = self.envs_obj_pcd_buffer[self.envs_idx]
            env_num, joint_num, point_num, point_dim = pcd_buffer.shape
            pcd_buffer = pcd_buffer.view(-1, point_num, point_dim)
            pcd_buffer = pcd_buffer[range(pcd_buffer.shape[0]), self.joint_idx_buff.view(-1)]
            pcd_buffer = pcd_buffer.reshape(env_num, joint_num, point_dim)
            pcd_buffer = pcd_buffer[env_ids]
            joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids][env_ids]

            joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode][env_ids]
            valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode][env_ids]
            joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
            pcd_buffer_view = pcd_buffer.view(-1, 3)
            pcd_buffer_view[valid_joint_contact_choice.view(-1)] = joints_contact[valid_joint_contact_choice.view(-1)]

            height_map = self.envs_heightmap[env_ids]
            contact_type = self.contact_type[env_ids]
            contact_valid = self.contact_valid[env_ids]
            contact_direction = self.contact_direction[env_ids]
        
            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset[env_ids]
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset[env_ids]
            tar_dir = self.tar_dir[env_ids]

        tar_rot = root_states.new_zeros([root_states.shape[0],4])
        tar_rot[:, 3] = 1
        tar_vel = root_states.new_zeros([root_states.shape[0],3])
        tar_ang_vel = root_states.new_zeros([root_states.shape[0],3])
        obs, self.local_height_map, self.rotated_mesh_pos = compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, tar_rot, tar_vel, tar_ang_vel, contact_type, contact_valid, contact_direction,
                                                                                 self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir, self.w_last)
        
        char_root_state = self.get_humanoid_root_states()
        target = self.stand_point

        pcd_buffer = self.envs_obj_pcd_buffer[self.envs_idx]
        joint_pos_buffer = self.new_rigid_body_pos[..., self._strike_body_ids, :]

        joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode]
        valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode]
        joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
        pcd_buffer_view = pcd_buffer.view(-1, pcd_buffer.shape[-2], 3)
        pcd_buffer_view[valid_joint_contact_choice.view(-1)] = joints_contact[valid_joint_contact_choice.view(-1)][:, None]
        pcd_buffer = pcd_buffer_view.reshape(pcd_buffer.shape)

        self.rew_buf[:], self.location_diff_buf[:], self.joint_diff_buff[:], self.joint_idx_buff[:], self.tar_dir[:, :2] = compute_contact_reward(target, char_root_state,
                                                                                 pcd_buffer, joint_pos_buffer,
                                                                                 self._prev_root_pos,
                                                                                 self.dt, self.contact_type, self.contact_valid, self.contact_direction, self.w_last)
        
        self.unihsi_obs[env_ids] = obs


    def compute_reset(self):
        super().compute_reset()
        # calcute reset conditions
        success = (self.location_diff_buf < 0.1) & ~self.big_force
        contact_type_steps = self.contact_type_step[self.scene_for_env, self.step_mode]
        contact_valid_steps = self.contact_valid_step[self.scene_for_env, self.step_mode]
        fulfill = ((contact_valid_steps & \
                     (((contact_type_steps) & (self.joint_diff_buff < 0.3)) | (((~contact_type_steps) & (self.joint_diff_buff >= 0.1))))) \
                        | (~contact_valid_steps))& (success[:, None])
        fulfill = torch.all(fulfill, dim=-1)
        self.still = self.still_buf>10 & fulfill
        self.big_force = (self._contact_forces.abs()>10000).sum((-2,-1))>0

        self.reset_buf[:], self._terminate_buf[:], self.still_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self._rigid_body_vel, self.still_buf, fulfill, self.big_force)



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





#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, tar_rot, tar_vel, tar_ang_vel, contact_type, contact_valid, contact_direction, 
                                human_in_mesh, origin_root_pos, local_scale, height_map_pcd, mesh_pos, tar_dir, w_last):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = rotations.quat_rotate(heading_rot, local_tar_pos, w_last)
    local_tar_vel = rotations.quat_rotate(heading_rot, tar_vel, w_last)
    local_tar_ang_vel = rotations.quat_rotate(heading_rot, tar_ang_vel, w_last)

    local_tar_rot = rotations.quat_mul(heading_rot, tar_rot, w_last)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot, w_last)

    mesh_pos = mesh_pos[None] + (root_pos -human_in_mesh)[:, None]
    mesh_dist = (mesh_pos - root_pos[:, None]).reshape(-1, 3)
    heading_rot_for_height = root_rot[:, None].repeat(1, local_scale*local_scale, 1).reshape(-1, 4)
    heading_rot_for_height[:, :2] = 0
    heading_rot_norm = torch.sqrt(1 - heading_rot_for_height[:, 3]*heading_rot_for_height[:, 3])
    heading_rot_for_height[:, 2] = heading_rot_for_height[:, 2] * heading_rot_norm / torch.abs(heading_rot_for_height[:, 2])
    rotated_mesh_dist = rotations.quat_rotate(heading_rot_for_height, mesh_dist, w_last).reshape(-1, local_scale*local_scale, 3)
    rotated_mesh_pos = rotated_mesh_dist + root_pos[:, None]
    rotated_mesh_origin_pos = rotated_mesh_dist + origin_root_pos[:, None]
    dist = rotated_mesh_origin_pos[..., :2][:, None] - height_map_pcd[..., :2][:, :, None]
    sum_dist = torch.sum(dist * dist, dim=-1)
    shape = sum_dist.shape
    sum_dist = sum_dist.permute(1,0,2).reshape(shape[1], -1)
    min_idx = sum_dist.argmin(0)
    dist_min = sum_dist[min_idx, range(min_idx.shape[0])]
    valid_mask = dist_min < 0.05
    height_map = height_map_pcd.reshape(-1, 3)[min_idx, 2]
    height_map[~valid_mask] = 0.0
    height_map = height_map.reshape(shape[0], shape[2]).float()


    contact_direction = contact_direction.reshape(-1, 45)
    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, tar_dir, contact_type, contact_valid, height_map], dim=-1)

    local_target_pos = pcd_buffer - joint_pos_buffer
    local_target_pos_r = rotations.quat_rotate(heading_rot[:, None].repeat(1, local_target_pos.shape[1], 1).view(-1, 4), local_target_pos.view(-1, 3), w_last)
    local_target_pos_r = local_target_pos_r.reshape(local_target_pos.shape)
    local_target_pos_r *= contact_valid[..., None]
    obs = torch.cat([obs, local_target_pos_r.view(obs.shape[0], -1), contact_direction], dim=-1)

    return obs, height_map, rotated_mesh_pos

def compute_contact_reward(target, root_state, pcd_buffer, joint_pos_buffer, 
                           prev_root_pos, dt, contact_type, contact_valid, contact_direction, w_last):
    dist_threshold = 0.2

    pos_err_scale = 0.5
    vel_err_scale = 2.0
    near_pos_err_scale = 10

    tar_speed = 1.0
    
    root_pos = root_state[..., 0:3]
    root_rot = root_state[..., 3:7]

    contact_type = contact_type.float()

    near_pos_reward_buf = []
    min_pos_idx_buf = []
    near_pos_err_min_buf = []
    for i in range(pcd_buffer.shape[1]):
        near_pos_diff = pcd_buffer[:, i] - joint_pos_buffer[:, i][:, None]
        near_pos_err = torch.sum(near_pos_diff * near_pos_diff, dim=-1)
        near_pos_err_min, min_pos_idx = near_pos_err.min(-1)
        near_pos_reward = torch.exp(-near_pos_err_scale * near_pos_err_min)
        near_pos_reward_contact = near_pos_reward * contact_type[:,i] + (1-near_pos_reward) * (1-contact_type[:,i])

        reward_dir = torch.nn.functional.normalize(-near_pos_diff[range(near_pos_diff.shape[0]), min_pos_idx], dim=-1) * contact_direction[:,i]
        reward_dir = reward_dir.sum(-1)
        reward_dir[contact_direction[:,i].sum(1)==0] = 1
        reward_dir[reward_dir<0] = 0
        reward_dir[~contact_type[:,i].bool()] = 1
        contact_w = (1 - near_pos_reward_contact) / (2 - near_pos_reward_contact - reward_dir + 1e-4)
        dir_w = (1 - reward_dir) / (2 - near_pos_reward_contact - reward_dir + 1e-4)
        near_pos_err_min[reward_dir<0.5] += 1 # not fullfill dir
        near_pos_reward_contact = contact_w * near_pos_reward_contact + dir_w * reward_dir

        near_pos_reward_contact[~contact_valid[:, i]] = 1
        near_pos_reward_buf.append(near_pos_reward_contact)
        min_pos_idx_buf.append(min_pos_idx)
        near_pos_err_min_buf.append(near_pos_err_min)
    near_pos_err_min_buf = torch.stack(near_pos_err_min_buf, 1)
    near_pos_reward_buf = torch.stack(near_pos_reward_buf, 0)
    min_pos_idx_buf = torch.stack(min_pos_idx_buf, 1)
    near_pos_reward_w = (1 - near_pos_reward_buf) / (pcd_buffer.shape[1] - near_pos_reward_buf.sum(0) + 1e-4)

    near_pos_reward = (near_pos_reward_w * near_pos_reward_buf).sum(0)

    facing_target = (contact_valid.sum(-1) == 1) & (contact_valid[:, -1] | contact_valid[:, -4])
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = rotations.quat_rotate(heading_rot, facing_dir, w_last)
    
    target_pcd = pcd_buffer[range(pcd_buffer.shape[0]), -1, min_pos_idx_buf[:, -1]]
    pos_diff = target_pcd - root_pos
    tar_dir = torch.nn.functional.normalize(pos_diff[..., 0:2], dim=-1)
    tar_dir[~facing_target] = facing_dir[~facing_target][..., 0:2]
    obj_facing_reward = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    obj_facing_reward = torch.clamp_min(obj_facing_reward, 0.0)

    obj_facing_reward_w = (1-obj_facing_reward) / (2-obj_facing_reward-near_pos_reward + 1e-4)
    near_pos_reward_w = (1-near_pos_reward) / (2-obj_facing_reward-near_pos_reward + 1e-4)
    
    near_pos_reward = obj_facing_reward_w * obj_facing_reward + near_pos_reward_w * near_pos_reward

    pos_diff = target - root_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    dist_mask = pos_err < dist_threshold

    tar_dir = torch.nn.functional.normalize(pos_diff[..., 0:2], dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0
    vel_reward[dist_mask] = 1

    heading_rot = torch_utils.calc_heading_quat(root_rot, w_last)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = rotations.quat_rotate(heading_rot, facing_dir, w_last)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    facing_reward[dist_mask] = 1

    pos_reward_w = (1-pos_reward) / (3-pos_reward-vel_reward-facing_reward + 1e-4)
    vel_reward_w = (1-vel_reward) / (3-pos_reward-vel_reward-facing_reward + 1e-4)
    face_reward_w = (1-facing_reward) / (3-pos_reward-vel_reward-facing_reward + 1e-4)

    far_reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    not_walking = contact_valid.sum(-1)>0

    # dist_mask = pos_err < dist_threshold
    reward = far_reward
    reward[not_walking] = near_pos_reward[not_walking]
    pos_err[not_walking] = 0 # once success, keep success
    # reward[dist_mask] = reward_near[dist_mask]

    return reward, pos_err, near_pos_err_min_buf, min_pos_idx_buf, tar_dir
    
# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, _rigid_body_vel, still_buf, fulfill, big_force):
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

        # tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        #strike_body_force = contact_buf[:, strike_body_id, :]
        #strike_body_has_contact = torch.any(torch.abs(strike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        # tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)
        
        # has_failed = torch.logical_or(has_fallen, tar_fail)
        has_failed = has_fallen

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
    
    still_buf[_rigid_body_vel.abs().sum(-1).max(-1)[0]<0.6] += 1
    still_buf[_rigid_body_vel.abs().sum(-1).max(-1)[0]>0.6] = 0
    # print(_rigid_body_vel.abs().sum(-1).max(-1))

    # print(still_buf)

    terminated = torch.where(big_force, torch.ones_like(reset_buf), terminated) # terminate when force is too big (could cause peneration)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    reset = torch.where((still_buf>10) & fulfill, torch.ones_like(reset_buf), reset)
    
    return reset, terminated, still_buf