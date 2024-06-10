from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import PromptEncoder, TwoWayTransformer, SwinTransformer, MaskDecoder
from matplotlib import pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from visual_sampler.sampler_v2 import build_shape_sampler
from visual_sampler.config import cfg


#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='?',
    # required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='/exports/lkeb-hpc-data/MedSAM/MedSAM_Laptop/segs_res/segs_scribble/',
    help='directory to save the prediction',
)
parser.add_argument(
    '-lite_medsam_checkpoint_path',
    type=str,
    default="?/Swin_LiteMedSAM.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    default=True,
    action='store_true',
    help='whether to save the overlay image'
)

parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlays/overlay_scribble',
    help='directory to save the overlay image'
)

args = parser.parse_args()
num_points = 4
data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)

lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256

def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
    
    def forward(self, image, points, boxes, masks, tokens):
        image_embedding, fs = self.image_encoder(image)
        with torch.no_grad():
            boxes = torch.as_tensor(boxes, dtype=torch.float32, device=image.device)
            if len(boxes.shape) == 2:
                boxes = boxes[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
            tokens=tokens,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            fs,
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

def resize_scribble_mask(mask, new_size, original_size):
    masks = mask[..., None, :new_size[0], :new_size[1]]
    masks =  F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

    return masks.squeeze()


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     

def show_points(points, ax):
    points = points.numpy()
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, color='yellow', s=15) 

def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box, ratio

def get_points_256_v0(box, gt2D):
    num_points = 4
    gt2D = np.mean(gt2D, axis=-1)
    if len(box)==1:
        x_min, y_min, x_max, y_max = box[0]
    else:
        x_min, y_min, x_max, y_max = box

    try:
        bounder_shiftx = np.random.randint(int((x_max-x_min)/5), int(2*(x_max-x_min)/4)-1, (1,))
    except:
        bounder_shiftx = 0
    try:
        bounder_shifty = np.random.randint(int((y_max-y_min)/5), int(2*(y_max-y_min)/4)-1, (1,))
    except:
        bounder_shifty = 0
    
    mid_x = int((x_min+x_max)//2)
    mid_y = int((y_min+y_max)//2)
    x_min = int(x_min+bounder_shiftx)
    x_max = int(x_max-bounder_shiftx)
    y_min = int(y_min+bounder_shifty)
    y_max = int(y_max-bounder_shifty)
    cl = [[y_min, mid_y, x_min, mid_x], [mid_y,y_max,x_min,mid_x], [mid_y,y_max, mid_x,x_max], [y_min,mid_y, mid_x,x_max]]

    coords = []
    for i in range(num_points):
        gt2D_tmp = np.zeros((256, 256))
        gt2D_tmp[cl[i][0]:cl[i][1], cl[i][2]:cl[i][3]] = gt2D[cl[i][0]:cl[i][1], cl[i][2]:cl[i][3]]
        y_indices, x_indices = np.where(gt2D_tmp > 0)
        if y_indices.size==0:
            coords.append([mid_x, mid_y])
        else:
            x_point = np.random.choice(x_indices)
            y_point = np.random.choice(y_indices)
            coords.append([x_point, y_point])
    coords = np.array(coords).reshape(num_points, 2)
    coords = torch.tensor(coords).float()
    return coords


def get_points_256(box, gt2D):
    gt2D = np.mean(gt2D, axis=-1)
    if len(box)==1:
        x_min, y_min, x_max, y_max = box[0]
    else:
        x_min, y_min, x_max, y_max = box
    mid_x = int((x_min+x_max)//2)
    mid_y = int((y_min+y_max)//2)
    try:
        bounder_shiftx = np.random.randint(int((x_max-x_min)/5), int(2*(x_max-x_min)/5), (1,))
    except:
        bounder_shiftx = 0
    try:
        bounder_shifty = np.random.randint(int((y_max-y_min)/5), int(2*(y_max-y_min)/5), (1,))
    except:
        bounder_shifty = 0

    x_min = int(x_min+bounder_shiftx)
    x_max = int(x_max-bounder_shiftx)
    y_min = int(y_min+bounder_shifty)
    y_max = int(y_max-bounder_shifty)

    coords = []
    gt2D_tmp = np.zeros((256, 256))
    gt2D_tmp[y_min:y_max, x_min:x_max] = gt2D[y_min:y_max, x_min:x_max]
    for i in range(num_points):
        y_indices, x_indices = np.where(gt2D_tmp > 0)
        if y_indices.size==0:
            coords.append([mid_x, mid_y])
        else:
            x_point = np.random.choice(x_indices)
            y_point = np.random.choice(y_indices)
            coords.append([x_point, y_point])
    coords = np.array(coords).reshape(num_points, 2)
    coords = torch.tensor(coords).float()
    return coords


def get_scribble_256(box, gt2D):
    gt2D = np.mean(gt2D, axis=-1)
    shape_sampler = build_shape_sampler(cfg)

    if len(box)==1:
        x_min, y_min, x_max, y_max = box[0]
    else:
        x_min, y_min, x_max, y_max = box

    try:
        bounder_shiftx = np.random.randint(int((x_max-x_min)/8), int((x_max-x_min)/6), (1,))
    except:
        bounder_shiftx = 0
    try:
        bounder_shifty = np.random.randint(int((y_max-y_min)/8), int((y_max-y_min)/6), (1,))
    except:
        bounder_shifty = 0

    x_min = int(x_min+bounder_shiftx)
    x_max = int(x_max-bounder_shiftx)
    y_min = int(y_min+bounder_shifty)
    y_max = int(y_max-bounder_shifty)
    gt2D_tmp = np.zeros((256, 256))
    gt2D_tmp[y_min:y_max, x_min:x_max] = gt2D[y_min:y_max, x_min:x_max]
    gt2D_tmp = np.uint8(gt2D_tmp>0)
    gt2D_tmp[gt2D_tmp>0] = 1

    masks = shape_sampler(gt2D_tmp).squeeze().unsqueeze(0).numpy()
    return torch.tensor(masks).float()


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, fs, points_256, box_256, scribble_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    points_256 = points_256[None, ...]
    labels_torch = torch.ones(points_256.shape[0]).long()
    labels_torch = labels_torch.unsqueeze(1).expand(-1, num_points)
    point_prompt = (points_256, labels_torch.to(device))
    scribble = scribble_256[None, ...].to(device)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = point_prompt,
        boxes = box_torch,
        masks = scribble,
        tokens=None,
    )

    low_res_logits, iou = medsam_model.mask_decoder(
        fs,
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg, iou

medsam_lite_image_encoder = SwinTransformer()

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
    transformer=TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    ),
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(lite_medsam_checkpoint["model"])
medsam_lite_model.to(device)
medsam_lite_model.eval()


def MedSAM_infer_npz_2D(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding, fs = medsam_lite_model.image_encoder(img_256_tensor)

    point_func = get_points_256
    
    points = []
    scribbles = []
    for idx, box in enumerate(boxes, start=1):
        box256, ratio = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        points256 = point_func(box256, img_256_padded)
        points256 = points256.to(device)
        scribble_256 = get_scribble_256(box256, img_256_padded)
        medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, fs, points256, box256, scribble_256, (newh, neww), (H, W))
        segs[medsam_mask>0] = idx
        s_mask = resize_scribble_mask(scribble_256, (newh, neww), (H, W))
        points.append(points256.cpu()/ratio)
        scribbles.append(s_mask.cpu().numpy())
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
    
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

    # visualize image, mask and bounding box
    if save_overlay and "Microscope" not in npz_name:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_points(points[i], ax[1])
            show_mask(scribbles[i].astype(np.uint8), ax[1])
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


def MedSAM_infer_npz_3D(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    # not used in this demo because it treats each slice independently
    # spacing = npz_data['spacing'] 
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    point_func = get_points_256

    inference_time = []
    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)
            
            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            
            with torch.no_grad():
                start_time = time()
                image_embedding, fs = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
                inference_time.append(time()-start_time)
            if z == z_middle:
                box_256, _ = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg256 = resize_longest_side(pre_seg)
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256, _ = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))

            points256 = point_func(box_256, img_256)
            points256 = points256.to(device)
            scribble_256 = get_scribble_256(box_256, img_256)
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, fs, points256, box_256, scribble_256, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                start_time = time()
                image_embedding, fs = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
                inference_time.append(time()-start_time)

            pre_seg = segs_3d_temp[z+1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg256 = resize_longest_side(pre_seg)
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                scale_256 = 256 / max(H, W)
                box_256 = mid_slice_bbox_2d * scale_256
            points256 = point_func(box_256, img_256)
            points256 = points256.to(device)
            scribble_256 = get_scribble_256(box_256, img_256)
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, fs, points256, box_256, scribble_256, [new_H, new_W], [H, W])
            
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
    print("inference time:", sum(inference_time))
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )            

    # visualize image, mask and bounding box
    if save_overlay and "Microscope" not in npz_name:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    
    img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith('3D'):
            MedSAM_infer_npz_3D(img_npz_file)
        else:
            MedSAM_infer_npz_2D(img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)

