import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path

from src.utils.train_util import instantiate_from_config

import open3d as o3d
import Imath
import OpenEXR
from src.utils.camera_util import get_zero123plus_input_cameras, get_relative_transformations

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        image_dir='rendering_zero123plus',
        validation=False,
        sample=4096,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = image_dir

        with open(os.path.join(root_dir, meta_fname)) as f:
            paths = json.load(f)
        # paths = []
        # for k in lvis_dict.keys():
        #     paths.extend(lvis_dict[k])
        self.paths = paths

        self.sample = sample
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[-30:] # used last 16 as validation
        else:
            self.paths = self.paths[:-30]
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_pc(self, path):
        """
            Points are normalized by
                bbox_min, bbox_max = scene_bbox()
                scale = 1 / np.linalg.norm(bbox_max - bbox_min)

        """
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)

        if points.shape[0] > self.sample:
            # Downsample to self.sample points
            indices = np.random.choice(points.shape[0], self.sample, replace=False)
            points = points[indices]
        else:
            # Pad with zeros if fewer points
            pad_size = self.sample - points.shape[0]
            pad_points = np.zeros((pad_size, 3))
            points = np.vstack([points, pad_points])

        points = torch.from_numpy(points).contiguous().float()  # (N, 3)

        return points

    def __getitem__(self, index):
        while True:
            image_path = os.path.join(self.root_dir, self.image_dir, self.paths[index])

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            depth_list = []
            pc_list = []
            try:
                for idx in range(7):
                    img, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bkg_color)
                    depth, alpha = self.load_im(os.path.join(image_path, '%03d_colored_depth.png' % idx), bkg_color)
                    points = self.load_pc(os.path.join(image_path, '%03d_pc.ply' % idx))
                    
                    img_list.append(img)
                    depth_list.append(depth)
                    pc_list.append(points)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        imgs = torch.stack(img_list, dim=0).float()
        depths = torch.stack(depth_list, dim=0).float()
        points = torch.stack(pc_list, dim=0).float()

        data = {
            'cond_imgs': imgs[0],           # (3, H, W)
            'cond_depths': depths[0],           # (3, H, W)
            'cond_pcs': points[0],           # (N, 3)
            'target_imgs': imgs[1:],        # (6, 3, H, W)
        }
        return data


class ObjaverseDataPlanar(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        image_dir='rendering_zero123plus',
        planar_dir='rendering_zero123planar',
        fov=30,
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = image_dir
        self.planar_dir = planar_dir
        self.fov = fov

        with open(os.path.join(root_dir, meta_fname)) as f:
            paths = json.load(f)
        # paths = []
        # for k in lvis_dict.keys():
        #     paths.extend(lvis_dict[k])
        self.paths = paths
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[-16:] # used last 16 as validation
        else:
            self.paths = self.paths[:-16]
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def load_exr_depth(self, path):
        # Open the EXR file
        exr_file = OpenEXR.InputFile(path)

        # Get header and image size
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Define the pixel type (32-bit float)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        # Read the depth channel (assuming 'R' if single-channel depth map)
        channel_name = list(header['channels'].keys())[0]  # Get the first channel (e.g., 'R')
        depth_data = np.frombuffer(exr_file.channel(channel_name, pt), dtype=np.float32)

        # Reshape to 2D depth map
        depth_map = depth_data.reshape((height, width))

        return depth_map

    def __getitem__(self, index):
        while True:
            image_path = os.path.join(self.root_dir, self.image_dir, self.paths[index])
            planar_image_path = os.path.join(self.root_dir, self.planar_dir, self.paths[index])

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            planar_img_list = []
            planar_pose_list = []
            planar_depth_list = []
            try:
                for idx in range(7):
                    img, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bkg_color)
                    img_list.append(img)

                for idx in range(6):
                    planar_img, alpha = self.load_im(os.path.join(planar_image_path, '%03d.png' % idx), bkg_color)
                    planar_img_list.append(planar_img)

                    # planar_img_depth = cv2.imread(os.path.join(planar_image_path, '%03d_depth.png' % idx), cv2.IMREAD_UNCHANGED)
                    planar_img_depth = self.load_exr_depth(os.path.join(planar_image_path, '%03d_depth.exr' % idx))
                    planar_depth_list.append(planar_img_depth)

                    cond_RT = np.load(os.path.join(planar_image_path, '%03d.npy' % idx))
                    planar_pose_list.append(cond_RT)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        # The last four items are intrinsics
        img_poses = get_zero123plus_input_cameras(batch_size=1, radius=2.0, fov=30.0, return_numpy=True)  # (6, 4, 4)
        planar_poses = np.stack(planar_pose_list, axis=0)  # (6, 4, 4)

        # Compute scales from the depth maps
        planar_depth_ndarray = np.stack(planar_depth_list, axis=0)
        masked_planar_depth_ndarray = planar_depth_ndarray.copy()
        # Replace infinity depth values (65504.0) with NaN
        masked_planar_depth_ndarray[masked_planar_depth_ndarray == 65504.0] = np.nan
        # Compute the 95th percentile while ignoring NaN values
        scales = np.nanquantile(masked_planar_depth_ndarray, q=0.2, axis=(1, 2))

        # Compute the relative poses Erel (6, 6, 4, 4)
        relative_poses = get_relative_transformations(img_poses, planar_poses, scales)
        relative_poses = relative_poses.flatten()
        fov_tensor = torch.Tensor([self.fov]).float()
        T = torch.concat([relative_poses, fov_tensor])   # (577)

        imgs = torch.stack(img_list, dim=0).float()
        planar_imgs = torch.stack(planar_img_list, dim=0).float()
        # Clamp depth to (0, 5)
        planar_depth = torch.from_numpy(planar_depth_ndarray).unsqueeze(1).float().clamp(0, 5)

        data = {
            'T': T,                                  # (577)
            'cond_depth': planar_depth,              # (6, 1, H, W)
            'cond_imgs': planar_imgs,                # (6, 3, H, W)
            'target_imgs': imgs[1:],                 # (6, 3, H, W)
        }
        return data