export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan24 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan24 --extract_mesh_only True
