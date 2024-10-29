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


def reshape_MR(img):

    original_shape = img.shape
    sorted_axes = np.argsort(original_shape)
    new_img = img.transpose(sorted_axes)

    return new_img


class NpzDataset_Scribble(Dataset):

    def __init__(self,
                 data_root,
                 points=True,
                 masks=True,
                 image_size=256,
                 bbox_shift=5,
                 data_aug=True):

        self.data_root = data_root

        subfolder_npz = glob.glob(join(self.data_root, "*/*.npz"),
                                  recursive=True)

        subsubfolder_npz = glob.glob(join(self.data_root, "*/*/*.npz"),
                                     recursive=True)

        self.file_paths = subfolder_npz + subsubfolder_npz

        assert len(self.file_paths) == 108714

        print("#" * 20)
        print("Total number of images: {0:.2f}K".format(
            len(self.file_paths) / 1e3))
        print("#" * 20)

        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        self.points = points
        self.masks = masks
        self.shape_sampler = build_shape_sampler(cfg)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):

        npz = np.load(self.file_paths[index], 'r', allow_pickle=True)
        img_name = basename(self.file_paths[index]).split(".")[0]
        
        gts = npz['gts']
        img = npz['imgs']

        # special case for MR_totalseg
        if "MR_totalseg" in img_name:
            img = reshape_MR(img)
            gts = reshape_MR(gts)


        if len(gts.shape) > 2:  ## 3D image
            i = random.randint(0, gts.shape[0] - 1)
            img = img[i, :, :]
            gts = gts[i, :, :]
            img_3c = np.repeat(img[:, :, None], 3, axis=-1)  # (H, W, 3)
            img_resize = self.resize_longest_side(img_3c)
        else:
            if len(img.shape) < 3:
                img_3c = np.repeat(img[:, :, None], 3, axis=-1)
            else:
                img_3c = img
            img_resize = self.resize_longest_side(img_3c)

        gts = np.uint16(gts)

        #####
        # Resize
        img_resize = (img_resize - img_resize.min()) / np.clip(
            img_resize.max() - img_resize.min(), a_min=1e-8,
            a_max=None)  # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1))  # (3, 256, 256)
        assert np.max(img_padded) <= 1.0 and np.min(
            img_padded) >= 0.0, 'image should be normalized to [0, 1]'
        
        
        label_ids = np.unique(gts)
        label_ids = label_ids.tolist()
        
        try:
            label_ids.remove(0)
            label_id = random.choice(label_ids)
            gt2D_original = np.uint8(gts == label_id) 
            gt2D = cv2.resize(
                gt2D_original,
                (img_resize.shape[1], img_resize.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            gt2D = self.pad_image(gt2D)

        except:
            return self.__getitem__(random.randint(0,len(self)-1))
        

        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))

        try:
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
        except:
            
            return self.__getitem__(random.randint(0,len(self)-1))

        if self.points:
            mid_x = (x_min + x_max) // 2
            mid_y = (y_min + y_max) // 2
            cl = [[y_min, mid_y, x_min, mid_x], [mid_y, y_max, x_min, mid_x],
                  [mid_y, y_max, mid_x, x_max], [y_min, mid_y, mid_x, x_max]]
            coords = []
            for i in range(4):
                gt2D_tmp = np.zeros((H, W))
                gt2D_tmp[cl[i][0]:cl[i][1],
                         cl[i][2]:cl[i][3]] = gt2D[cl[i][0]:cl[i][1],
                                                   cl[i][2]:cl[i][3]]
                y_indices, x_indices = np.where(gt2D_tmp > 0)
                if y_indices.size == 0:
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
            "image":
            torch.tensor(img_padded).float(),
            "gt2D":
            torch.tensor(gt2D[None, :, :]).long(),
            "coords":
            coords,
            "masks":
            torch.tensor(masks).float(),
            "bboxes":
            torch.tensor(bboxes[None, None, ...]).float(),
            "image_name":
            img_name,
            "new_size":
            torch.tensor(np.array([img_resize.shape[0],
                                   img_resize.shape[1]])).long(),
            "original_size":
            torch.tensor(np.array([img.shape[0], img.shape[1]])).long()
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
        if len(image.shape) == 3:  ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else:  ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded




