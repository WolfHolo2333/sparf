export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_bear
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_clock
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_dog
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_durian
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_jade
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_man
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_sculpture
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_stone