CUDA_VISIBLE_DEVICES=3 python phys_anim/train_agent.py \
+exp=ase \
experiment_name=ase_sas_test_gym \
+robot=sword_and_shield \
motion_file=data/motions/ase_motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
+backbone=isaacgym +opt=wandb num_envs=3584