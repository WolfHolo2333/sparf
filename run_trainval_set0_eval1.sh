export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan24 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan24 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan37 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan37 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan40 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan40 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan55 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan55 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan63 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan63 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan65 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan65 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan69 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan69 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan83 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan83 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan97 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan97 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan105 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan105 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan106 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan106 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan110 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan110 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan114 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan114 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan118 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan118 --extract_mesh_only True

python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan122 --save_pose_only True
python run_trainval.py joint_pose_nerf_training/dtu sparf --train_sub 3 --scene scan122 --extract_mesh_only True

python eval_result_process.py
