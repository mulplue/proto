# smpl walk
CUDA_VISIBLE_DEVICES=0 python phys_anim/train_agent.py \
+exp=amp \
experiment_name=amp_smpl_walk_lab \
+robot=smpl \
motion_file=data/motions/smpl_humanoid_walk.npy \
+backbone=isaaclab +opt=wandb

# smpl path follower
CUDA_VISIBLE_DEVICES=0 python phys_anim/train_agent.py \
+exp=path_follower \
experiment_name=amp_smpl_path_follower_lab_3584 \
+robot=smpl \
motion_file=data/motions/smpl_humanoid_walk.npy \
+backbone=isaaclab +opt=wandb num_envs=3584

# smpl path following
CUDA_VISIBLE_DEVICES=1 python phys_anim/train_agent.py \
+exp=steering \
experiment_name=amp_smpl_steering_lab_3584 \
+robot=smpl \
motion_file=data/motions/smpl_humanoid_walk.npy \
+backbone=isaaclab +opt=wandb num_envs=3584


"""ASE task + ASE motion"""
# sword and shield location
CUDA_VISIBLE_DEVICES=3 python phys_anim/train_agent.py \
+exp=location \
experiment_name=amp_sas_location_lab \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaaclab +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=2 python phys_anim/train_agent.py \
+exp=heading \
experiment_name=amp_sas_heading_lab \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaaclab +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=1 python phys_anim/train_agent.py \
+exp=reach \
experiment_name=amp_sas_reach_lab \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaaclab +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=0 python phys_anim/train_agent.py \
+exp=amp \
experiment_name=amp_sas_walk_lab \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy \
+backbone=isaaclab +opt=wandb num_envs=3584