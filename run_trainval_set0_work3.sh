export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan105
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan106
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan110
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan114
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan118
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan122