import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin import SwinTransformer
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transform import TwoWayTransformer

class MedSAM_Lite(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

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
    
# medsam_lite_image_encoder = SwinTransformer()

# medsam_lite_prompt_encoder = PromptEncoder(
#     embed_dim=256,
#     image_embedding_size=(64, 64),
#     input_image_size=(256, 256),
#     mask_in_chans=16
# )

# medsam_lite_mask_decoder = MaskDecoder(
#     num_multimask_outputs=3,
#     transformer=TwoWayTransformer(
#         depth=2,
#         embedding_dim=256,
#         mlp_dim=2048,
#         num_heads=8,
#     ),
#     transformer_dim=256,
#     iou_head_depth=3,
#     iou_head_hidden_dim=256,
# )

# medsam_lite_model = MedSAM_Lite(
#     image_encoder = medsam_lite_image_encoder,
#     mask_decoder = medsam_lite_mask_decoder,
#     prompt_encoder = medsam_lite_prompt_encoder
# )