"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """


'''
4xsubsampled version of DTU dataset, using rectified images
Resolution: 300x400
Scenes: 124 scenes
Poses: 49 poses

Loading DTU data in DVR format processed by pixelNeRF

Download dataset from:
https://github.com/sxyu/pixel-nerf

Data convention description not really followed:
https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/FAQ.md

- pixelNeRF uses whole P matrix, world_mat_i in this case is P = K [R|t] !
- IDR also uses DVR convention but different scale matrix

Since the camera matrices are the same for each scene, just load it once for 49 poses

Camera space follows OpenCV convention, x(right), y(down), z(in)
Cameras
#      0-4
#    10 - 5
#   11  -  18
#  27  xx   19
# 28    x    38
#48     -     39


'''
import os
import imageio
import numpy as np
import torch
from PIL import Image
import cv2
import re
from typing import Any, List, Dict, Tuple
import logging

from source.datasets.base import Dataset
from source.utils.euler_wrapper import prepare_data

class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    def skew_symmetric(self,w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

def read_pfm(filename: str) -> Tuple[np.ndarray, float]:
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


class DTUDatasetPixelNerf(Dataset):
    def __init__(self, args: Dict[str, Any], split: str, scenes: str='', **kwargs):
        super().__init__(args, split)
        
        # paths are from args.env
        self.base_dir = args.env.dtu

        if hasattr(args, 'copy_data') and args.copy_data:
            prepare_data(args.env.dtu_depth_tar, args.env.untar_dir)
            prepare_data(args.env.dtu_mask_tar, args.env.untar_dir)
            
        self.depth_dir = args.env.dtu_depth
        self.dtu_mask_path = args.env.dtu_mask
        
        # This is to rescale the poses and the depth maps. 
        # Here, I hard-coded 1./300. because for all the scenes I considered, the 
        # scaling factor "norm_scale" (see L.236) was always equal to [300., 300., 300.]
        # If this is not the case, this needs to be modified to be equal to 1./norm_scale. 
        # One should also verify that the scaling makes the depth maps consistent with the poses. 
        self.scaling_factor = 1./300.  

        self.near_depth = 1.2
        self.far_depth = 5.2

        self.scene = scenes
        self.pose=Pose()
        self.lie=Lie()

        print(f"Loading scene {scenes} from DTU Dataset from split {self.split}...")

        scene_path = os.path.join(self.base_dir, self.scene)
        _, rgb_files, intrinsics, poses, poses_noise = self.load_scene_data(scene_path)
        self.all_poses_c2w = poses

        # split the files into train and test
        if self.args.dtu_split_type == 'set0':
            train_idx = [0,1,2]
            test_idx = [0,1,2]
            split_indices = {'test': test_idx, 'train': train_idx}
        elif self.args.dtu_split_type == 'pixelnerf':
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            split_indices = {'test': test_idx, 'train': train_idx}
        elif self.args.dtu_split_type == 'all':
            idx = list(np.arange(49))
            split_indices = {'test': idx, 'train': idx}
        elif self.args.dtu_split_type == 'pixelnerf_reduced_testset':
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13, 24, 30, 41, 47, 43, 29, 45,
                        34, 33]
            test_idx = [1, 2, 9, 10, 11, 12, 14, 15, 23, 26, 27, 31, 32, 35, 42, 46]
            split_indices = {'test': test_idx, 'train': train_idx}
        else:
            all_indices = np.arange(len(rgb_files))
            split_indices = {
                'test': all_indices[all_indices % self.args.dtuhold == 0],
                'train': all_indices[all_indices % self.args.dtuhold != 0],
            }

        indices_train = split_indices['train']
        indices_test = split_indices['test']

        # train split
        if self.args.train_sub is not None:
            # here, will take the subset of 1, 3 or 9
            indices_train = indices_train[:self.args.train_sub]

        if self.args.val_sub is not None:
            indices_test = indices_test[:self.args.val_sub]

        train_masks_files, test_masks_files = self._load_mask_paths(self.scene, indices_train, indices_test)
        # self.training_masks_files, self.test_masks_files

        train_rgb_files = np.array(rgb_files)[indices_train]
        train_intrinsics = np.array(intrinsics)[indices_train]
        train_poses = np.array(poses)[indices_train]
        train_poses_noise = np.array(poses_noise)[indices_train]

        test_rgb_files = np.array(rgb_files)[indices_test]
        test_intrinsics = np.array(intrinsics)[indices_test]
        test_poses = np.array(poses)[indices_test]
        test_poses_noise = np.array(poses_noise)[indices_test]
        
        # rendering split 
        if 'train' in self.split:
            render_rgb_files = train_rgb_files
            render_intrinsics = train_intrinsics
            render_poses = train_poses
            render_poses_noise = train_poses_noise
            img_indices = indices_train
            render_mask_files = train_masks_files
        else:
            render_rgb_files = test_rgb_files
            render_intrinsics = test_intrinsics
            render_poses = test_poses
            render_poses_noise = test_poses_noise
            img_indices = indices_test
            render_mask_files = test_masks_files

        self.render_rgb_files = render_rgb_files.tolist()
        self.render_poses_c2w = render_poses
        self.render_poses_c2w_noise = render_poses_noise
        self.render_intrinsics = render_intrinsics
        self.render_masks_files = render_mask_files
        self.render_img_id = img_indices

        print(f"In total there are {len(self.render_rgb_files)} images in this dataset")

    def invert_pose(self, pose):
        pose = self.pose(R=pose[:3, :3], t=pose[:3, 3])
        pose_inv = self.pose.invert(pose)
        return torch.cat(
            [pose_inv, torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=pose.device).view(1, 4)], -2)

    def load_K_Rt_from_P(self,P):
        """
        modified from IDR https://github.com/lioryariv/idr
        """
        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K/K[2,2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3,3] = (t[:3] / t[3])[:,0]

        return intrinsics, pose

    def load_scene_data(self, scene_path: str):
        """
        Args:
            scene_path: path to scene directory
        Returns: 
            list of file names, rgb_files, np.array of intrinsics (Nx4x4), poses (Nx4x4)
        """
        # will load all iamges and poses 

        img_path = os.path.join(scene_path, "image")

        if not os.path.isdir(img_path):
            raise FileExistsError(img_path)

        # all images
        file_names = [f.split(".")[0] for f in sorted(os.listdir(img_path))]
        rgb_files = [os.path.join(img_path, f) for f in sorted(os.listdir(img_path))]
        pose_indices = [int(os.path.basename(e)[:-4]) for e in rgb_files] # this way is safer than range

        camera_info = np.load(os.path.join(scene_path, "cameras_sphere.npz"))

        camera_noise_path = os.path.join(scene_path, "noises0.15.pt")
        assert os.path.exists(camera_noise_path)
        se3_noise = torch.load(camera_noise_path)

        intrinsics = []
        poses_c2w = []
        poses_c2w_noise = []

        for p in pose_indices:
            world_mat = camera_info[f"world_mat_{p}"] # Projection matrix 
            scale_mat = camera_info.get(f"scale_mat_{p}")

            P = world_mat @ scale_mat
            P = P[:3,:4]
            intrinsics_,pose_ = self.load_K_Rt_from_P(P)

            w2c = self.invert_pose(pose_)
            pose_base = self.pose(R=w2c[:3,:3], t=w2c[:3,3])

            w2c_noisy = self.pose.compose([self.lie.se3_to_SE3(se3_noise[p]), pose_base])
            w2c_noisy = torch.cat([w2c_noisy, torch.tensor([0,0,0,1], dtype=torch.float32, device=w2c.device).view(1,4)], -2).cpu().numpy()

            c2w_noisy = self.invert_pose(w2c_noisy)

            poses_c2w.append(pose_)
            poses_c2w_noise.append(c2w_noisy)
            intrinsics.append(intrinsics_)

        intrinsics = np.stack(intrinsics, axis=0)
        poses_c2w = np.stack(poses_c2w, axis=0)
        poses_c2w_noise = np.stack(poses_c2w_noise, axis=0)

        return file_names, rgb_files, intrinsics, poses_c2w, poses_c2w_noise

    def _load_mask_paths(self, scene: str, train_idx: List[int], test_idx: List[int]):
        """Load masks from disk."""
        masks = []

        mask_path = self.dtu_mask_path
        maskf_fn = lambda x: os.path.join(  # pylint: disable=g-long-lambda
            mask_path, scene, 'mask', f'{x:03d}.png')

        train_masks_files = [maskf_fn(i) for i in train_idx]
        test_masks_files = [maskf_fn(i) for i in test_idx]
        return train_masks_files, test_masks_files

    def read_depth(self, filename: str):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600) 

        depth_h *= self.scaling_factor
        # the depth is at the original resolution of the images
        return depth_h

    def get_all_noise_poses(self, args):
        # of the current split
        return torch.inverse(torch.from_numpy(self.render_poses_c2w_noise))[:, :3].float()  # (B, 3, 4)

    def get_all_camera_poses(self, args):
        # of the current split
        return torch.inverse(torch.from_numpy(self.render_poses_c2w))[:, :3].float()  # (B, 3, 4)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int)

        Returns:
            a dictionary for each image index containing the following elements: 
                * idx: the index of the image
                * rgb_path: the path to the RGB image. Will be used to save the renderings with a name. 
                * image: the corresponding image, a torch Tensor of shape [3, H, W]. The RGB values are 
                            normalized to [0, 1] (not [0, 255]). 
                * intr: intrinsics parameters, numpy array of shape [3, 3]
                * pose:  world-to-camera transformation matrix in OpenCV format, numpy array of shaoe [3, 4]
                * depth_range: depth_range, numpy array of shape [1, 2]
                * scene: scene name

                * depth_gt: ground-truth depth map, numpy array of shape [H, W]
                * valid_depth_gt: mask indicating where the depth map is valid, bool numpy array of shape [H, W]
                * fg_mask: foreground segmentation mask, bool numpy array of shape [1, H, W]

        """
        
        rgb_file = self.render_rgb_files[idx]
        render_pose_c2w = self.render_poses_c2w[idx]
        render_pose_w2c = np.linalg.inv(render_pose_c2w)
        render_intrinsics = self.render_intrinsics[idx]
        img_id = self.render_img_id[idx]
        scene = self.scene

        # read and handle the image to render
        rgb = imageio.imread(rgb_file)
        h, w = rgb.shape[:2]

        mask_file = self.render_masks_files[idx]
        if os.path.exists(mask_file):
            with open(mask_file, 'rb') as imgin:
                mask = np.array(Image.open(imgin), dtype=np.float32)[:, :, :3] / 255.
                mask = mask[:, :, 0]  # (H, W)
                mask = (mask == 1).astype(np.bool_)
        else:
            mask = np.ones_like(rgb[:, :, 0], np.bool_)  # (h, W)


        depth_filename = os.path.join(self.depth_dir, f'Depths/{scene}/depth_map_{img_id:04d}.pfm')
        if os.path.exists(depth_filename):
            depth_gt = self.read_depth(depth_filename)
        else:
            print(f'Could not find {depth_filename}')
            depth_gt = np.zeros((h, w), dtype=np.float32)

        rgb, render_intrinsics, depth_gt, mask = \
            self.preprocess_image_and_intrinsics(rgb, intr=render_intrinsics, 
            depth=depth_gt, mask=mask, channel_first=False)

        valid_depth_gt = depth_gt > 0.

        if self.args.mask_img:
            # we do not want a black background
            # instead white is better
            mask_torch = torch.from_numpy(mask).unsqueeze(-1).float()
            rgb = rgb * mask_torch + 1 - mask_torch
            valid_depth_gt = valid_depth_gt & mask

        near_depth = self.near_depth * (1-self.args.increase_depth_range_by_x_percent)
        far_depth = self.far_depth * (1+self.args.increase_depth_range_by_x_percent)
        depth_range = torch.tensor([near_depth, far_depth], dtype=torch.float32)

        assert mask.shape[:2] == rgb.shape[:2]
        assert depth_gt.shape[:2] == rgb.shape[:2]
        assert valid_depth_gt.shape[:2] == rgb.shape[:2]
        ret =  {
            'idx': idx, 
            "rgb_path": rgb_file,
            'depth_gt': depth_gt, # (H, W)
            'fg_mask': np.expand_dims(mask, 0),  # numpy array, (1, H, W), bool
            'valid_depth_gt': valid_depth_gt,  # (H, W) 
            'image': rgb.permute(2, 0, 1), # torch tensor 3, self.H, self.W
            'intr': render_intrinsics[:3, :3].astype(np.float32),
            'pose':  render_pose_w2c[:3].astype(np.float32),   # # 3x4, world to camera
            "depth_range": depth_range,
            'scene': self.scene
        }
        return ret

