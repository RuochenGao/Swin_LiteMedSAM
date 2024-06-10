import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import torch
import torch.nn as nn
import time
import cv2
from transformers import CLIPTokenizer, CLIPTextModel
from os.path import join, exists, isfile, isdir, basename
import random
join = os.path.join

from visual_sampler.sampler_v2 import build_shape_sampler
from visual_sampler.config import cfg
import json

seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class NpyDataset_Scribble(Dataset): 
    def __init__(self, data_root, points=True, masks=True, texts=False, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        # CT_LungMasks_lung_047-003.npy only has the label of 0
        self.gt_path_files = glob.glob(join(self.gt_path, "*.npy"))
        print("#" * 20)
        print("Total number of images: {0:.2f}M".format(
            len(self.gt_path_files) / 1e6))
        print("#" * 20)
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        self.points = points
        self.masks = masks
        self.texts = texts
        self.shape_sampler = build_shape_sampler(cfg)
        
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        img = np.load(join(self.img_path, img_name), 'r', allow_pickle=True)
        if img.shape[-1]==1:
            img = np.repeat(img,3,axis=-1)
        img_resize = self.resize_longest_side(img)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True)
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = self.pad_image(gt)
        label_ids = np.unique(gt)[1:]
        
        try:
            # label_id = random.choice(label_ids.tolist())
            gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
        
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        instance = np.zeros_like(gt2D)
        instance[y_min:y_max, x_min:x_max] = 1
        
        if self.points:
            mid_x = (x_min+x_max)//2
            mid_y = (y_min+y_max)//2
            cl = [[y_min, mid_y, x_min, mid_x], [mid_y,y_max,x_min,mid_x], [mid_y,y_max, mid_x,x_max], [y_min,mid_y, mid_x,x_max]]
            coords = []
            for i in range(4):
                gt2D_tmp = np.zeros((H, W))
                gt2D_tmp[cl[i][0]:cl[i][1], cl[i][2]:cl[i][3]] = gt2D[cl[i][0]:cl[i][1], cl[i][2]:cl[i][3]]
                y_indices, x_indices = np.where(gt2D_tmp > 0)
                if y_indices.size==0:
                    coords.append([mid_x, mid_y])
                else:
                    x_point = np.random.choice(x_indices)
                    y_point = np.random.choice(y_indices)
                    coords.append([x_point, y_point])
            coords = np.array(coords).reshape(4, 2)
            coords = torch.tensor(coords).float()
        else:
            coords = None
            
        if self.masks:
            masks = self.shape_sampler(instance).squeeze().unsqueeze(0).numpy()
        else:
            masks = None

        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "coords": coords,
            "masks": torch.tensor(masks).float(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(),
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img.shape[0], img.shape[1]])).long()
        }
        
    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded