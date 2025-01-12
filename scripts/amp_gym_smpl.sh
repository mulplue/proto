"""Proto task + Proto motion + SMPL"""
# smpl walk
CUDA_VISIBLE_DEVICES=3 python phys_anim/train_agent.py \
+exp=amp \
experiment_name=amp_smpl_walk_gym_3584 \
+robot=smpl \
motion_file=data/motions/smpl_humanoid_walk.npy \
+backbone=isaacgym +opt=wandb num_envs=3584

# smpl path follower
CUDA_VISIBLE_DEVICES=3 python phys_anim/train_agent.py \
+exp=path_follower \
experiment_name=fixed_amp_smpl_path_follower_gym_3072 \
+robot=smpl \
motion_file=data/motions/smpl_humanoid_walk.npy \
+backbone=isaacgym +opt=wandb num_envs=3072

# smpl path steering
CUDA_VISIBLE_DEVICES=4 python phys_anim/train_agent.py \
+exp=steering \
experiment_name=fixed_amp_smpl_steering_gym_3072 \
+robot=smpl \
motion_file=data/motions/smpl_humanoid_walk.npy \
+backbone=isaacgym +opt=wandb num_envs=3072


"""Proto task + ASE motion + SMPL"""
CUDA_VISIBLE_DEVICES=7 python phys_anim/train_agent.py \
+exp=steering \
experiment_name=amp_smpl_steering_gym \
+robot=smpl \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=6 python phys_anim/train_agent.py \
+exp=path_follower \
experiment_name=amp_smpl_path_follower_gym \
+robot=smpl \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

"""ASE task + ASE motion + SMPL"""
CUDA_VISIBLE_DEVICES=7 python phys_anim/train_agent.py \
+exp=location \
experiment_name=amp_smpl_location_gym \
+robot=smpl \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=6 python phys_anim/train_agent.py \
+exp=heading \
experiment_name=amp_smpl_heading_gym \
+robot=smpl \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=4 python phys_anim/train_agent.py \
+exp=amp \
experiment_name=amp_smpl_walk_gym \
+robot=smpl \
motion_file=data/motions/ase_motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy \
+backbone=isaacgym +opt=wandb num_envs=3584

