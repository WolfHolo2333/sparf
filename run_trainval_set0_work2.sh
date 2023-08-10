export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan24
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan37
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan40
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan55
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan63
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan65
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan69
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan83
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene scan97