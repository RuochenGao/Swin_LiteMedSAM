import os
import glob
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import multiprocessing as mp
from tqdm import tqdm


def resize_longest_side(image, target_length=256):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def pad_image(image, target_length=256):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_length - h
    padw = target_length - w
    if len(image.shape) == 3:  ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:  ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded


def convert_npz_to_npy(npz_path):
    """
    Convert npz files to npy files for training

    Parameters
    ----------
    npz_path : str
        Name of the npz file to be converted
    """
    try:
        name = basename(npz_path).split(".npz")[0]
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        imgs = npz["imgs"]
        gts = npz["gts"]
        
        if len(gts.shape) > 2:  ## 3D image
        
            for i in range(imgs.shape[0]):
                img_i = imgs[i, :, :]
                gt_i = gts[i, :, :]
             
                img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)

                img_01 = (img_3c - img_3c.min()) / np.clip(
                    img_3c.max() - img_3c.min(), a_min=1e-8,
                    a_max=None)  # normalize to [0, 1], (H, W, 3)

                img_01 = np.array(img_01, dtype=np.float32)
                assert img_01.shape[:2] == gt_i.shape
                
                gt_i = gts[i, :, :]
                gt_i[gt_i > 255] = 0
                gt_i = np.uint8(gt_i)
                

                img_01 = img_01[:, :, :
                                1]  # only use the first channel, for CT MRI PET 3D gray images

                np.save(
                    join(npy_dir, "imgs",
                         name + "-" + str(i).zfill(3) + ".npy"), img_01)
                np.save(
                    join(npy_dir, "gts",
                         name + "-" + str(i).zfill(3) + ".npy"), gt_i)

        else:  ## 2D image
            if len(imgs.shape) < 3:
                img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
            else:
                img_3c = imgs

         

            img_01 = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8,
                a_max=None)  # normalize to [0, 1], (H, W, 3)
            
            
            img_01 = np.array(img_01, dtype=np.float32)
            assert img_01.shape[:2] == gts.shape
            
            # For microscopy images, the number of label is larger than 255
            gts[gts > 255] = 0
            gts = np.uint8(gts)   
            
            np.save(join(npy_dir, "imgs", name + ".npy"), img_01)
            np.save(join(npy_dir, "gts", name + ".npy"), gts)
            
    except Exception as e:
        print(e)
        print(npz_path)





if __name__ == "__main__":

    npz_dir = "MedSAM_Laptop/train_npz/"
    npy_dir = "MedSAM_Laptop/train_npy/"
    npz_paths = glob.glob(join(npz_dir, "**/*.npz"), recursive=True)

    print("npz file: ", len(npz_paths))
    num_workers = 32
    do_resize_256 = False  # whether to resize images and masks to 256x256
    makedirs(npy_dir, exist_ok=True)
    makedirs(join(npy_dir, "imgs"), exist_ok=True)
    makedirs(join(npy_dir, "gts"), exist_ok=True)

    with mp.Pool(num_workers) as p:
        r = list(
            tqdm(p.imap(convert_npz_to_npy, npz_paths), total=len(npz_paths)))
    
            
