CUDA_VISIBLE_DEVICES=2 python phys_anim/train_agent.py \
 +exp=full_body_tracker  \
 experiment_name=maskedmimic_fbt_smpl_walk_gym \
 +robot=smpl \
 motion_file=data/motions/smpl_humanoid_walk.npy \
 +backbone=isaacgym +opt=wandb num_envs=3072
