from isaacgym import gymapi, gymtorch  # type: ignore[misc]
import json
import torch
import numpy as np
import open3d as o3d

from isaac_utils import rotations
from phys_anim.envs.unihsi.common import BaseUnihsi
from phys_anim.envs.base_task.isaacgym import TaskHumanoid
 


class UnihsiHumanoid(BaseUnihsi, TaskHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):

        num_scenes = config.unihsi_params.num_scenes
        self.num_scenes_col = num_scenes
        self.num_scenes_row = num_scenes
        self.env_array = torch.arange(0, config.num_envs, device=device, dtype=torch.float)
        self.spacing = config.unihsi_params.env_spacing
        
        strike_body_names = config.unihsi_params.strike_body_names
        self.joint_num = len(strike_body_names)
        self.local_scale = 9
        self.local_interval = 0.2
        self.max_step_pool_number = 30

        sceneplan_path = config.unihsi_params.obj_file
        with open(sceneplan_path) as f:
            self.sceneplan = json.load(f)
        self.plan_items = self.sceneplan

        self.joint_name = ["pelvis", "left_hip", "left_knee", "left_foot", "right_hip", "right_knee", "right_foot", "torso", 
            "head", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
        self.joint_mapping = {"pelvis":0, "left_hip":1, "left_knee":2, "left_foot":3, "right_hip":4, "right_knee":5, "right_foot":6, "torso":7, 
            "head":8, "left_shoulder":9, "left_elbow":10, "left_hand":11, "right_shoulder":12, "right_elbow":13, "right_hand":14}


        super().__init__(config=config, device=device)
        # if not self.headless:
        #     self._build_marker_state_tensors()

    

    def create_envs(self, num_envs, spacing, num_per_row):
        self.spacing = spacing # env==2,3 have stupid bug

        self.env_scene_idx_row = (self.env_array % num_per_row % self.num_scenes_row).long()
        self.env_scene_idx_col = (self.env_array // num_per_row % self.num_scenes_col).long()
        self.scene_for_env = self.scene_idx[self.env_scene_idx_row, self.env_scene_idx_col]

        self.x_offset = (self.env_array % num_per_row % self.num_scenes_row) * spacing * 2 - (self.env_array % num_per_row) * spacing * 2
        self.y_offset = (self.env_array // num_per_row % self.num_scenes_col) * spacing * 2- (self.env_array // num_per_row) * spacing * 2

        # [num_envs, num_obj, num_part_sequence, num_pts. 3]
        self.envs_obj_pcd_buffer = self.obj_pcd_buffer.new_zeros([self.num_envs, self.obj_pcd_buffer.shape[2], self.obj_pcd_buffer.shape[3], self.obj_pcd_buffer.shape[4]])
        
        # self.envs_heightmap = self.height_map[self.scene_for_env].float()

        self.obj_rotate_matrix = self.obj_rotate_matrix.permute(2,3,0,1).float()
        self.scene_stand_point = torch.einsum("nmoe,neg->nmog", self.scene_stand_point[self.scene_for_env], self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])
        self.scene_stand_point[..., 0] += self.x_offset[:,None,None] + self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]
        self.scene_stand_point[..., 1] += self.y_offset[:,None,None] + self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]     
        
        self.envs_heightmap = self.height_map[self.scene_for_env].float()
        self.envs_heightmap[..., 0] += self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap[..., 1] += self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap[..., 2] += self.rand_dist_z[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap = torch.einsum("nae,neg->nag", self.envs_heightmap, self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])
        
        super().create_envs(num_envs, spacing, num_per_row)


    def draw_task(self):
        self.gym.clear_lines(self.viewer)
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        starts = self.new_rigid_body_pos[0][self._strike_body_ids][self.contact_valid[0]]
        ends = self.envs_obj_pcd_buffer[0][range(15), self.joint_idx_buff[0]][self.contact_valid[0]]

        joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids][0]
        joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode][0]
        valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode][0]
        joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
        ends[valid_joint_contact_choice.view(-1)[self.contact_valid[0]]] = joints_contact[valid_joint_contact_choice.view(-1)]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()
        # verts = verts[7:9]
        cols = cols.repeat(verts.shape[0], axis=0)

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts
            curr_verts = curr_verts.reshape(-1, 6)
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)


        starts = self._humanoid_root_states[..., 0:3]
        # ends = self.pcd_buffer[0].mean(1)
        ends = self.stand_point
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)


    # create ground plane
    def create_ground_plane(self):
        self._create_mesh_ground()
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_mesh_ground(self):
        self.plan_number = len(self.sceneplan)
        min_mesh_dict = self._load_mesh()
        pcd_list = self._load_pcd(min_mesh_dict)

        self._get_pcd_parts(pcd_list)

        _x = np.arange(0, self.local_scale)
        _y = np.arange(0, self.local_scale)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        mesh_grid = np.stack([x, y, y], axis=-1) # 3rd dim meaningless
        self.mesh_pos = torch.from_numpy(mesh_grid).to(self.device) * self.local_interval
        self.humanoid_in_mesh = torch.tensor([self.local_interval*(self.local_scale-1)/4, self.local_interval*(self.local_scale-1)/2, 0]).to(self.device)

    def process_contact(self, label_dict, pcd, obj, steps, step_number, plan_id):

        direction_mapping = {
            "up":[0,0,1],
            "down":[0,0,-1],
            "left":[0,1,0],
            "right":[0,-1,0],
            "front":[1,0,0],
            "back":[-1,0,0],
            "none":[0,0,0]
        }
        for step_idx, step in enumerate(steps):
            contact_type_step = np.zeros(15)
            contact_valid_step = np.zeros(15)
            joint_pairs = np.zeros(15)
            joint_pairs_valid = np.zeros(15)
            contact_direction_step = np.zeros((15,3))

            for pair in step:
                obj_id = pair[0][-3:]
                label = label_dict[obj_id]['label']
                label_mapping = label_dict[obj_id]['label_mapping']
                if 'stand_point' in obj[obj_id].keys():
                    stand_point = obj[obj_id]['stand_point']
                else:
                    stand_point = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
                obj_pcd = pcd[int(obj_id)*10000:(int(obj_id)+1)*10000]
                if pair[1] != 'none' and pair[1] not in self.joint_name:
                    part_pcd = self._get_obj_parts(pair[0], [pair[1]], label_mapping, label, obj_pcd, stand_point)
                    joint_number = self.joint_mapping[pair[2]]
                    self.obj_pcd_buffer[plan_id, step_idx, joint_number] = part_pcd[0]
                    contact_type_step[joint_number] = 1 if pair[3] == 'contact' else 0
                    contact_valid_step[joint_number] = 1
                    contact_direction_step[joint_number] = direction_mapping[pair[4]]
                elif pair[1] in self.joint_name:
                    joint_number = self.joint_mapping[pair[2]]
                    target_joint_number = self.joint_mapping[pair[1]]
                    joint_pairs[joint_number] = target_joint_number
                    joint_pairs_valid[joint_number] = 1
                    contact_direction_step[joint_number] = direction_mapping[pair[4]]
            
            self.scene_stand_point[plan_id, step_idx] = torch.tensor(stand_point).float()
            self.contact_type_step[plan_id, step_idx] = torch.tensor(contact_type_step)
            self.contact_valid_step[plan_id, step_idx] = torch.tensor(contact_valid_step)
            self.contact_direction_step[plan_id, step_idx] = torch.tensor(contact_direction_step) 
            self.joint_pairs[plan_id, step_idx] = torch.tensor(joint_pairs)
            self.joint_pairs_valid[plan_id, step_idx] = torch.tensor(joint_pairs_valid)

    def _load_mesh(self):

        mesh_vertices_list = []
        mesh_triangles_list = []

        min_mesh_dict = dict()
        for plans in self.plan_items:
            plan = self.plan_items[plans]
            objs = plan['obj']
            min_mesh_dict[plans] = dict()

            l = 0
            pn = 0
            mesh_vertices = np.zeros([0, 3]).astype(np.float32)
            mesh_triangles = np.zeros([0, 3]).astype(np.uint32)
            for obj_id in objs:
                obj = objs[obj_id]
                pid = obj['id']
                mesh = o3d.io.read_triangle_mesh('data/unihsi_data/data/partnet/'+pid+'/models/model_normalized.obj')
                for r in obj['rotate']:
                    R = mesh.get_rotation_matrix_from_xyz(r)
                    mesh.rotate(R, center=(0, 0, 0))
                mesh.scale(obj['scale'], center=mesh.get_center())
                mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
                mesh.translate((0,0,-mesh_vertices_single[:, 2].min()))
                mesh.translate(obj['transfer']) #  not collision with init human
                # mesh.translate(obj['multi_obj_offset'])
                mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
                mesh_triangles_single = np.asarray(mesh.triangles).astype(np.uint32)

                min_x_mesh, min_y_mesh, min_z_mesh = mesh_vertices_single[:,0].min(), mesh_vertices_single[:,1].min(), mesh_vertices_single[:,2].min()
                min_mesh_dict[plans][obj_id] = [min_x_mesh, min_y_mesh, min_z_mesh]

                mesh_vertices = np.concatenate([mesh_vertices, mesh_vertices_single], axis=0)
                mesh_triangles = np.concatenate([mesh_triangles, mesh_triangles_single], axis=0)
                mesh_triangles[l:] += pn
                l = mesh_triangles.shape[0]
                pn = mesh_vertices.shape[0]

            mesh_vertices_list.append(mesh_vertices)
            mesh_triangles_list.append(mesh_triangles)

        

        obj_idx = np.random.randint(0, self.plan_number, (self.num_scenes_row, self.num_scenes_col))
        obj_rotate = np.random.rand(self.num_scenes_row, self.num_scenes_col) * 360.0
        obj_rotate_matrix = np.array([[np.cos(np.radians(obj_rotate)), -np.sin(np.radians(obj_rotate)), obj_rotate*0],
                                    [np.sin(np.radians(obj_rotate)), np.cos(np.radians(obj_rotate)), obj_rotate*0],
                                    [obj_rotate*0, obj_rotate*0, obj_rotate*0+1]])

        self.obj_idx = torch.from_numpy(obj_idx).to(self.device)
        self.obj_rotate_matrix = torch.from_numpy(obj_rotate_matrix).to(self.device)

        dist_max = 2
        dist_min = 1
        rand_dist_x = (dist_max - dist_min) * np.random.rand(self.num_scenes_row, self.num_scenes_col) + dist_min
        rand_dist_y = (dist_max - dist_min) * np.random.rand(self.num_scenes_row, self.num_scenes_col) + dist_min
        rand_dist_x[0] = 0
        rand_dist_y[0] = 0
        self.rand_dist_x = torch.from_numpy(rand_dist_x).to(self.device)
        self.rand_dist_y = torch.from_numpy(rand_dist_y).to(self.device)
        rand_dist_z = np.random.rand(self.num_scenes_row, self.num_scenes_col) * 1.2 - 0.6
        change_height = 0
        rand_dist_z = rand_dist_z * change_height
        self.rand_dist_z = torch.from_numpy(rand_dist_z).to(self.device)


        scene_idx = np.random.randint(0, self.plan_number, (self.num_scenes_row, self.num_scenes_col))
        self.scene_idx = torch.from_numpy(scene_idx).to(self.device)

        for i in range(self.num_scenes_row):
            for j in range(self.num_scenes_col):
                mesh_vertices = mesh_vertices_list[self.scene_idx[i,j]]
                mesh_triangles = mesh_triangles_list[self.scene_idx[i,j]]
                mesh_vertices_offset = mesh_vertices.copy()
                mesh_vertices_offset = VaryPoint(mesh_vertices_offset, 'Z', obj_rotate[i,j]).astype(np.float32)
                mesh_vertices_offset[:, 0] += self.spacing *2 * i + rand_dist_x[i, j]
                mesh_vertices_offset[:, 1] += self.spacing *2 * j + rand_dist_y[i, j]
                mesh_vertices_offset[:, 2] += rand_dist_z[i, j]

                tm_params = gymapi.TriangleMeshParams()
                tm_params.nb_vertices = mesh_vertices_offset.shape[0]
                tm_params.nb_triangles = mesh_triangles.shape[0]
                self.gym.add_triangle_mesh(self.sim, mesh_vertices_offset.flatten(order='C'),
                                        mesh_triangles.flatten(order='C'),
                                        tm_params)
        
        return min_mesh_dict
    
    def _load_pcd(self, min_mesh_dict):

        trans_mat_path = 'data/unihsi_data/data/partnet/chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json'
        with open(trans_mat_path, 'r') as fcc_file:
            trans_mat = fcc_file.read()
        trans_mat = json.loads(trans_mat)

        pcds = dict()
        for plans in self.plan_items:
            plan = self.plan_items[plans]
            objs = plan['obj']

            pcd_multi = []
            for obj_id in objs:
                obj = objs[obj_id]
                pid = obj['id']
                pcd = o3d.io.read_point_cloud("data/unihsi_data/data/partnet/"+pid+"/point_sample/sample-points-all-pts-label-10000.ply")

                if pid == '11570' or pid == "11873" or pid == "4376" or pid == "5861":
                    pcd.scale(0.5, center=pcd.get_center())
                else:
                    matrix = np.array(trans_mat[pid]['transmat']).reshape(4,4)
                    tmp = matrix[0].copy()
                    matrix[0] = matrix[2]
                    matrix[2] = tmp
                    matrix = np.linalg.inv(matrix)
                    pcd.transform(matrix)

                for r in obj['rotate']:
                    R = pcd.get_rotation_matrix_from_xyz(r)
                    pcd.rotate(R, center=(0, 0, 0))
                pcd.scale(obj['scale'], center=pcd.get_center())

                if pid == '11570':
                    R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi))
                    pcd.rotate(R, center=(0, 0, 0))
                

                pcd = np.asarray(pcd.points).astype(np.float32())

                max_y = pcd[:, 1].max()
                pcd[:, 1] = max_y - pcd[:, 1] # flip

                min_x_pcd, min_y_pcd, min_z_pcd = pcd[:,0].min(), pcd[:,1].min(), pcd[:,2].min()
                min_x_mesh, min_y_mesh, min_z_mesh = min_mesh_dict[plans][obj_id]
                pcd[:,0] += min_x_mesh-min_x_pcd 
                pcd[:,1] += min_y_mesh-min_y_pcd 
                pcd[:,2] += min_z_mesh-min_z_pcd
                pcd_multi.append(pcd)
            pcd_multi = np.concatenate(pcd_multi, axis=0)

            pcds[plans] = pcd_multi
        
        return pcds

    def _get_pcd_parts(self, pcd_list):
        # self.valid_joints_mask = torch.zeros([self.obj_number, 2, self.joint_num], dtype=bool, device=self.device) # hard code

        self.height_map = []
        self.obj_pcd_buffer = torch.zeros([self.plan_number, self.max_step_pool_number, 15, 200, 3])
        self.contact_type_step = torch.zeros([self.plan_number, self.max_step_pool_number, 15])
        self.contact_valid_step = torch.zeros([self.plan_number, self.max_step_pool_number, 15])
        self.contact_direction_step = torch.zeros([self.plan_number, self.max_step_pool_number, 15, 3])
        self.joint_pairs = torch.zeros([self.plan_number, self.max_step_pool_number, 15])
        self.joint_pairs_valid = torch.zeros([self.plan_number, self.max_step_pool_number, 15])        
        self.scene_stand_point = torch.zeros([self.plan_number, self.max_step_pool_number, 4, 3])        
        self.max_steps = torch.zeros(self.plan_number).int()
        self.contact_pairs = []

        for idx, plan_id in enumerate(self.plan_items):
            obj = self.plan_items[plan_id]['obj']
            contact_pairs = self.plan_items[plan_id]['contact_pairs']
            pcd = pcd_list[plan_id]
            step_number = len(contact_pairs)

            label_dict = dict()
            for obj_item in obj:
                obj_id = obj[obj_item]['id']
                label, label_mapping = self._get_part_labels(obj_id)
                label_dict[obj_item] = {'label': label, 'label_mapping': label_mapping}

            self.process_contact(label_dict, pcd, obj, contact_pairs, step_number, idx)
            heigh_pcd = get_height_map(pcd, HEIGHT_MAP_DIM=20)
            self.height_map.append(torch.from_numpy(heigh_pcd).to(self.device))
            self.max_steps[idx] = step_number
            self.contact_pairs.append(contact_pairs)


            

        self.height_map = torch.stack(self.height_map,0).to(self.device)
        self.scene_stand_point =  self.scene_stand_point.to(self.device)
        self.contact_type_step = self.contact_type_step.to(self.device).bool()
        self.contact_valid_step = self.contact_valid_step.to(self.device).bool()
        self.contact_direction_step = self.contact_direction_step.to(self.device).long()
        self.joint_pairs = self.joint_pairs.to(self.device).long()
        self.joint_pairs_valid = self.joint_pairs_valid.to(self.device).bool()
        self.contact_valid_step = self.contact_valid_step | self.joint_pairs_valid  # TODO add joint contact type
        self.contact_type_step = self.contact_type_step | self.joint_pairs_valid
        self.obj_pcd_buffer = self.obj_pcd_buffer.to(self.device)
        self.max_steps = self.max_steps.to(self.device)

    def _get_part_labels(self, partnet_id):

        if not isinstance(partnet_id, list):
            partnet_id = [partnet_id]

        result_dict_full = dict()
        offset = 0
        labels = []
        for pid in partnet_id:
            label_file = "data/unihsi_data/data/partnet/"+pid+"/point_sample/sample-points-all-label-10000.txt"
            label = load_label(label_file) + offset
            result_file = "data/unihsi_data/data/partnet/"+pid+"/result.json"
            with open(result_file, 'r') as fcc_file:
                result_file = fcc_file.read()
            result = json.loads(result_file)
            result_dict = dict()
            get_leaf_node(result_dict, result, offset)
            result_dict_full.update(result_dict)
            offset += 100 # hard code
            labels.append(label)
        labels = np.concatenate(labels, axis=0)
        
        return labels, result_dict_full

    def _get_obj_parts(self, object, contact_parts, label_mapping, label, pcd, stand_point):

        max_x, min_x, max_y, min_y = pcd[:, 0].max(), pcd[:, 0].min(), pcd[:, 1].max(), pcd[:, 1].min()   

        obj_pcd_buffer = []
        for p in contact_parts:
            if 'floor' in p:
                out_part_pcd = torch.rand(30000,3).cuda()
                out_part_pcd[:,0] = out_part_pcd[:,0] * ((max_x+0.5)-(min_x-0.5)) + min_x-0.5
                out_part_pcd[:,1] = out_part_pcd[:,1] * ((max_y+0.5)-(min_y-0.5)) + min_y-0.5
                out_part_pcd[:,2] = out_part_pcd[:,2] * 0.08
                if object[:3] == "bed":
                    mask = (out_part_pcd[:,1]>max_y+0.2)
                elif  object[:3] == "chair":
                    mask = (out_part_pcd[:,0]>max_x)
                else:
                    mask = ((out_part_pcd[:,0]>max_x) | (out_part_pcd[:,0]<min_x)) | ((out_part_pcd[:,1]>max_y) | (out_part_pcd[:,1]<min_y))
                out_part_pcd = out_part_pcd[mask]
            else:
                idx = label_mapping[p]
                part_pcd = pcd[label==idx]
                max_x, min_x, max_y, min_y, max_z, min_z = part_pcd[:,0].max(), part_pcd[:,0].min(), part_pcd[:,1].max(), part_pcd[:,1].min(), part_pcd[:,2].max(), part_pcd[:,2].min()
            
                out_part_pcd = part_pcd[(part_pcd[:,0] <= max((max_x-0.2), (max_x-min_x)/5*4+min_x)) & (part_pcd[:,0] >= min(min_x+0.2, (max_x-min_x)/5*1+min_x)) &
                                    (part_pcd[:,1] <= max((max_y-0.2), (max_y-min_y)/5*4+min_y)) & (part_pcd[:,1] >= min(min_y+0.2, (max_y-min_y)/5*1+min_y))] # filter edge
                out_part_pcd = torch.from_numpy(out_part_pcd).to(self.device)
            idx = farthest_point_sample(out_part_pcd[None], 200)
            out_part_pcd = out_part_pcd[idx]
            obj_pcd_buffer.append(out_part_pcd)
        obj_pcd_buffer = torch.cat(obj_pcd_buffer, 0)

        return obj_pcd_buffer



def load_label(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        label = np.array([int(line) for line in lines], dtype=np.int32)
        return label

def get_leaf_node(dic, results, offset):
    for i in range(len(results)):
        result = results[i]
        if 'children' in result.keys():
            get_leaf_node(dic, result['children'], offset)
        else:
            dic[result['name']+str(result['id'])] = result['id'] + offset

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def get_height_map(points: np.ndarray, HEIGHT_MAP_DIM: int=16):
    """ Load region meshes of a scenes

    Args:
        points: scene point cloud
        HEIGHT_MAP_DIM: height map dimension
    
    Return:
        Return the floor height map and axis-aligned scene bounding box
    """

    ## compute floor height map
    minx, miny = points[:, 0].min(), points[:, 1].min()
    maxx, maxy = points[:, 0].max(), points[:, 1].max()

    interval_x = (maxx-minx)/HEIGHT_MAP_DIM
    interval_y = (maxy-miny)/HEIGHT_MAP_DIM
    voxel_idx_x = (points[:, 0]-minx) // interval_x
    voxel_idx_y = (points[:, 1]-miny) // interval_y

    height_map2d = np.zeros((HEIGHT_MAP_DIM,HEIGHT_MAP_DIM))
    for i in range(HEIGHT_MAP_DIM):
        for j in range(HEIGHT_MAP_DIM):
            mask = (voxel_idx_x==i)&(voxel_idx_y==j)
            if mask.sum()==0:
                pass
            else:
                height_map2d[j,i] = points[mask, 2].max()

    x = np.linspace(minx, maxx, HEIGHT_MAP_DIM)
    y = np.linspace(miny, maxy, HEIGHT_MAP_DIM)
    xx, yy = np.meshgrid(x, y)
    # pos2d = np.concatenate([yy[..., None], xx[..., None]], axis=-1)
    pos2d = np.concatenate([xx[..., None], yy[..., None]], axis=-1)

    height_pcd = np.concatenate([pos2d.reshape(-1,2),height_map2d.reshape(-1,1)], axis=-1)

    return height_pcd

def VaryPoint(data, axis, degree):
    xyzArray = {
        'X': np.array([[1, 0, 0],
                  [0, np.cos(np.radians(degree)), -np.sin(np.radians(degree))],
                  [0, np.sin(np.radians(degree)), np.cos(np.radians(degree))]]),
        'Y': np.array([[np.cos(np.radians(degree)), 0, np.sin(np.radians(degree))],
                  [0, 1, 0],
                  [-np.sin(np.radians(degree)), 0, np.cos(np.radians(degree))]]),
        'Z': np.array([[np.cos(np.radians(degree)), -np.sin(np.radians(degree)), 0],
                  [np.sin(np.radians(degree)), np.cos(np.radians(degree)), 0],
                  [0, 0, 1]])}
    newData = np.dot(data, xyzArray[axis])
    return newData