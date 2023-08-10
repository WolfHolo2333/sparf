export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_bear --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_clock --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_dog --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_durian --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_jade --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_man --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_sculpture --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf_wo_depth_cons_loss --train_sub 3 --scene bmvs_stone --save_pose_only True