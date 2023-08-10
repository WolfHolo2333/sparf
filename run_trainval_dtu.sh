export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan24
# python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan24 --extract_mesh_only True