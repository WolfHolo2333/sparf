export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_bear --extract_mesh_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_clock --extract_mesh_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_dog --extract_mesh_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_durian --extract_mesh_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_jade --extract_mesh_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_man --extract_mesh_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_sculpture --extract_mesh_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene bmvs_stone --extract_mesh_only True
