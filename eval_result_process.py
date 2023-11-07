from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import torch
import glob
from tqdm import tqdm
import trimesh

def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).to(R.device)

    R00 = R[..., 0,0]
    R01 = R[..., 0, 1]
    R02 = R[..., 0, 2]
    R10 = R[..., 1, 0]
    R11 = R[..., 1, 1]
    R12 = R[..., 1, 2]
    R20 = R[..., 2, 0]
    R21 = R[..., 2, 1]
    R22 = R[..., 2, 2]

    q[...,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[..., 1]=(R21-R12)/(4*q[:,0])
    q[..., 2] = (R02 - R20) / (4 * q[:, 0])
    q[..., 3] = (R10 - R01) / (4 * q[:, 0])
    return q

def tensor_to_traj(poses):
    return PoseTrajectory3D(
        positions_xyz=poses[:, :3, 3],
        orientations_quat_wxyz=rot_to_quat(poses[:, :3, :3]),
        timestamps=np.arange(len(poses)).astype(np.float32)
    )

def traj_to_tensor(traj):
    return torch.from_numpy(np.stack(traj.poses_se3)).float()

def save_evo_pose(filename, traj):
    # traj = tensor_to_traj(poses)
    file_interface.write_tum_trajectory_file(filename, traj)

def load_evo_pose(filename):
    return file_interface.read_tum_trajectory_file(filename)

def main():
    result_path = "./workspace_test/joint_pose_nerf_training/dtu/subset_3"
    sequence_fnames = sorted(glob.glob(os.path.join(result_path, 'scan*')))

    for i in tqdm(range(len(sequence_fnames))):
        print("processing %s" % sequence_fnames[i])
        subject_name = os.path.basename(sequence_fnames[i])
        save_path = "./result/%s" % subject_name

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pose_path = f"{sequence_fnames[i]}/sparf/poses/pose_trained_c2w.npy"
        mesh_path = f"{sequence_fnames[i]}/sparf/mesh/mesh.obj"
    
        pose_trained_c2w = np.load(pose_path)
        pose_trained_c2w = torch.tensor(pose_trained_c2w)
        traj = tensor_to_traj(pose_trained_c2w)
        save_evo_pose(f"{save_path}/est_poses.txt", traj)

        mesh = trimesh.load_mesh(mesh_path)
        mesh.vertices[:, [0, 1]] = mesh.vertices[:, [1, 0]]
        mesh.export(f"{save_path}/mesh.ply", file_type='ply')

main()