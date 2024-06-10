import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import torch
import time
join = os.path.join

class Distill_Dataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.target_length = 256
        self.image_size = 256
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.embed_path = "?/train_embedding_1024"

        self.gt_path_files = glob.glob(join(self.gt_path, "*.npy"))
        print("#" * 20)
        print("Total number of images: {0:.2f}M".format(
            len(self.gt_path_files) / 1e6))
        print("#" * 20)

    def __getitem__(self, index):
        # load npy image (h, w, 1), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img = np.load(join(self.img_path, img_name))
        with h5py.File(join(self.embed_path, f"{img_name.split('.')[0]}.h5"), 'r') as f:
            embedding = np.array(f["image_data"])

        # resize to (3, 256, 256)
        # img_256 = self.m2_pre_img(img)
        img_256 = self.pre_img(img)
        
        embedding = torch.tensor(embedding)
        return (
            img_256,
            embedding
        )
        
    def __len__(self):
        return len(self.gt_path_files)
    
    def pre_img(self, img, image_size=256):
        if img.shape[-1]==1:
            img = np.repeat(img,3,axis=-1)
        img_resize = self.resize_longest_side(img)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
        img_padded = self.pad_image(img_resize)
        img_padded = np.transpose(img_padded, (2, 0, 1))
        return torch.tensor(img_padded).float()
    
    def m2_pre_img(self, image_data, image_size=256):
        transform1 = transforms.Compose([
            transforms.ToTensor(), # normalize to [0.0,1.0]
            transforms.Resize([image_size, image_size], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            ]
        )
        
        resize_img_torch = transform1(image_data)
        if resize_img_torch.shape[0] == 1:
            resize_img_torch = resize_img_torch.repeat(3, 1, 1)
        
        return resize_img_torch
    
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


