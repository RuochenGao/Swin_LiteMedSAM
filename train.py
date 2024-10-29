import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from npz_dataset import NpzDataset_Scribble
from shutil import copyfile
from models import PromptEncoder, TwoWayTransformer, SwinTransformer, MaskDecoder, MaskDecoder_Prompt
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_root",
                        type=str,
                        default="?",
                        help="Path to the npy data root.")
    
    parser.add_argument('--task_name', type=str, default='Swin_LiteMedSAM', help='Task name')
    
    parser.add_argument("--pretrained_checkpoint",
                        type=str,
                        default=None,
                        help="Path to the pretrained Lite-MedSAM checkpoint.")
 
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="Path to the checkpoint to continue training.")
    
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./work_dir",
        help=
        "Path to the working directory where checkpoints and logs will be saved."
    )
    parser.add_argument('--data_aug',
                        action='store_true',
                        default=False,
                        help='use data augmentation during training')
    
    parser.add_argument("--num_epochs",
                        type=int,
                        default=20,
                        help="Number of epochs to train.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=8,
                        help="Number of workers for dataloader.")

    parser.add_argument(
        "--bbox_shift",
        type=int,
        default=5,
        help="Perturbation to bounding box coordinates during training.")
    
    
    parser.add_argument("--encoder_freeze", action="store_true", help="Whether to freeze the encoder.")
    
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    
    parser.add_argument("-weight_decay",
                        type=float,
                        default=0.001,
                        help="Weight decay.")
    parser.add_argument("-iou_loss_weight",
                        type=float,
                        default=1.0,
                        help="Weight of IoU loss.")
    parser.add_argument("-seg_loss_weight",
                        type=float,
                        default=1.0,
                        help="Weight of segmentation loss.")
    parser.add_argument("-ce_loss_weight",
                        type=float,
                        default=1.0,
                        help="Weight of cross entropy loss.")
    parser.add_argument("--sanity_check",
                        action="store_true",
                        default=True,
                        help="Whether to do sanity check for dataloading.")

    args = parser.parse_args()
    return args


def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='blue',
                      facecolor=(0, 0, 0, 0),
                      lw=2))


def show_points(points, ax):
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, color='red', s=10)


def cal_iou(result, reference):

    intersection = torch.count_nonzero(torch.logical_and(result, reference),
                                       dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference),
                                dim=[i for i in range(1, result.ndim)])

    iou = intersection.float() / union.float()

    return iou.unsqueeze(1)


def sanity_check_dataset(args):
    
    tr_dataset = NpzDataset_Scribble(args.data_root, data_aug=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    
    for step, batch in enumerate(tr_dataloader):
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        image = batch["image"]
        gt = batch["gt2D"]
        coords = batch["coords"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]
        mask = batch["masks"]

        axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        show_mask(mask[idx].cpu().squeeze().numpy(),
                  axs[0],
                  random_color=False)
        show_box(bboxes[idx].numpy().squeeze(), axs[0])
        show_points(coords[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        show_mask(mask[idx].cpu().squeeze().numpy(),
                  axs[1],
                  random_color=False)
        show_box(bboxes[idx].numpy().squeeze(), axs[1])
        show_points(coords[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        # set title
        axs[1].set_title(names_temp[idx])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(join(args.work_dir,
                         'Sanitycheck_DA.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()
        break


class MedSAM_Lite(nn.Module):

    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder,
                 encoder_freeze):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.encoder_freeze = encoder_freeze
        
        if self.encoder_freeze:
            print("Freezing the encoder.")
            encoder_weight_file = "?/Swin_encoder.pth"
            self.image_encoder.load_state_dict(
                torch.load(encoder_weight_file)["model"])
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            print("Not freezing the encoder.")

    def forward(self, image, points, boxes, masks, tokens):
        if self.encoder_freeze:
            with torch.no_grad():
                image_embedding, fs = self.image_encoder(
                    image)  # (B, 256, 64, 64)
        else:
            image_embedding, fs = self.image_encoder(image)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
            tokens=tokens,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            fs,
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    """
    batch_dict = {}
    for key in batch[0].keys():
        if key == "image_name":
            batch_dict[key] = [sample[key] for sample in batch]
        else:
            batch_dict[key] = torch.stack([sample[key] for sample in batch],
                                          dim=0)

    return batch_dict


if __name__ == "__main__":

    args = get_args()
    sanity_check_dataset(args)


    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    print(f"Run ID: {run_id}")
    

    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    makedirs(model_save_path, exist_ok=True)
    copyfile(__file__,
             join(model_save_path, run_id + "_" + os.path.basename(__file__)))

    device = torch.device("cuda")

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers

    medsam_lite_image_encoder = SwinTransformer()

    medsam_lite_prompt_encoder = PromptEncoder(embed_dim=256,
                                               image_embedding_size=(64, 64),
                                               input_image_size=(256, 256),
                                               mask_in_chans=16)

    medsam_lite_mask_decoder = MaskDecoder_Prompt(
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

    medsam_lite_model = MedSAM_Lite(image_encoder=medsam_lite_image_encoder,
                                    mask_decoder=medsam_lite_mask_decoder,
                                    prompt_encoder=medsam_lite_prompt_encoder,
                                    encoder_freeze=args.encoder_freeze)

    
    
    if (not os.path.exists(args.resume)) and isfile(
            args.pretrained_checkpoint):
        ## Load pretrained checkpoint if there's no checkpoint to resume from and there's a pretrained checkpoint
        print(
            f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
        medsam_lite_checkpoint = torch.load(args.pretrained_checkpoint,
                                            map_location="cpu")
        medsam_lite_model.load_state_dict(medsam_lite_checkpoint, strict=True)

    medsam_lite_model = medsam_lite_model.to(device)

    medsam_lite_model.train()

    print(
        f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}"
    )

    
    print('lr:', args.lr)
    
    optimizer = optim.AdamW(
        medsam_lite_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.8,
                                                        patience=5,
                                                        cooldown=0)
    seg_loss = monai.losses.DiceLoss(sigmoid=True,
                                     squared_pred=True,
                                     reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    iou_loss = nn.MSELoss(reduction='mean')

    train_dataset = NpzDataset_Scribble(data_root=args.data_root,
                                        data_aug=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    
    if os.path.exists(args.resume):
        print('Resuming from', args.resume)
        ckpt_folders = sorted(listdir(args.resume))
        ckpt_folders = [
            f for f in ckpt_folders
            if (f.startswith(args.task_name)
                and isfile(join(args.resume, f, 'medsam_lite_latest.pth')))
        ]
        print('*' * 20)
        print('existing ckpts in', args.resume, ckpt_folders)
        # find the latest ckpt folders
        time_strings = [
            f.split(args.task_name + '-')[-1] for f in ckpt_folders
        ]
        dates = [datetime.strptime(f, '%Y%m%d-%H%M') for f in time_strings]
        latest_date = max(dates)
        latest_ckpt = join(
            args.work_dir,
            args.task_name + '-' + latest_date.strftime('%Y%m%d-%H%M'),
            'medsam_lite_latest.pth')
        print('Loading from', latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location=device)
        medsam_lite_model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10

    train_losses = []
    epoch_times = []

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            coords = batch["coords"]
            masks = batch["masks"]
            optimizer.zero_grad()
            image, gt2D, boxes, coords, masks = image.to(device), gt2D.to(
                device), boxes.to(device), coords.to(device), masks.to(device)
            labels_torch = torch.ones(coords.shape[0]).long()
            labels_torch = labels_torch.unsqueeze(1).expand(-1, 4)
            labels_torch = labels_torch.to(device)
            point_prompt = (coords, labels_torch)
            logits_pred, iou_pred = medsam_lite_model(image, point_prompt,
                                                      boxes, masks, None)
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            mask_loss = l_seg + l_ce
            # mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
            with torch.no_grad():
                iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            loss = mask_loss + l_iou
            # loss = mask_loss + iou_loss_weight * l_iou
            epoch_loss[step] = loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}, loss: {loss.item():.4f}")

        epoch_end_time = time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)

        train_losses.append(epoch_loss_reduced)
        lr_scheduler.step(epoch_loss_reduced)

        model_weights = medsam_lite_model.state_dict()
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss_reduced,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_lite_latest.pth"))

        if epoch_loss_reduced < best_loss:
            print(
                f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
            best_loss = epoch_loss_reduced
            checkpoint["best_loss"] = best_loss

            torch.save(checkpoint, join(model_save_path,
                                        "medsam_lite_best.pth"))

        epoch_loss_reduced = 1e10

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].title.set_text("Dice + Binary Cross Entropy + IoU Loss")
        axes[0].plot(train_losses)
        axes[0].set_ylabel("Loss")
        axes[1].plot(epoch_times)
        axes[1].title.set_text("Epoch Duration")
        axes[1].set_ylabel("Duration (s)")
        axes[1].set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(join(model_save_path, "log.png"))
        plt.close()
