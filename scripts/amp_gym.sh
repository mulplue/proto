"""Proto task + Proto motion"""
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

"""Proto task + ASE motion"""
# sword and shield walk
CUDA_VISIBLE_DEVICES=6 python phys_anim/train_agent.py \
+exp=amp \
experiment_name=amp_sas_walk_gym_3584 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

# sword and shield path follower
CUDA_VISIBLE_DEVICES=5 python phys_anim/train_agent.py \
+exp=path_follower \
experiment_name=fixed_amp_sas_path_follower_gym_3072 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3072

# smpl path steering
CUDA_VISIBLE_DEVICES=4 python phys_anim/train_agent.py \
+exp=steering \
experiment_name=fixed_amp_sas_steering_gym_3072 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3072


"""ASE task + ASE motion"""
# sword and shield location
CUDA_VISIBLE_DEVICES=7 python phys_anim/train_agent.py \
+exp=location \
experiment_name=amp_sas_location_gym_3584 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=3 python phys_anim/train_agent.py \
+exp=heading \
experiment_name=amp_sas_heading_gym_3584 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=1 python phys_anim/train_agent.py \
+exp=reach \
experiment_name=amp_sas_reach_gym_3584 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

# # for test
# CUDA_VISIBLE_DEVICES=2 python phys_anim/train_agent.py \
# +exp=location \
# experiment_name=amp_sas_location_gym_3584 \
# +robot=sword_and_shield \
# motion_file=data/motions/amp_sword_and_shield_humanoid_walk.npy \
# +backbone=isaacgym +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=0 python phys_anim/train_agent.py \
+exp=location_t1 \
experiment_name=amp_sas_location_gym_3584_disc0.8 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584

CUDA_VISIBLE_DEVICES=2 python phys_anim/train_agent.py \
+exp=heading_t1 \
experiment_name=amp_sas_heading_gym_3584_disc0.8 \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584