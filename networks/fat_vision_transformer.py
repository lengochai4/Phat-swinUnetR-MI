# coding=utf-8
from __future__ import absolute_import, division, print_function
import copy
import logging
import torch
import torch.nn as nn

# Prefer relative import when used as a package; fallback for script usage
# Import linh hoạt
try:
    from .fat_swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
except Exception:  # pragma: no cover
    from fat_swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys  # type: ignore

# Ghi lại log thay vì print
logger = logging.getLogger(__name__)

# Chỉ import lớp SwinUnet, loại bỏ các lớp hoạt động khác trong file
__all__ = [
    "SwinUnet",
]


class SwinUnet(nn.Module):
    """
    Thin wrapper giữ nguyên cấu trúc “Swin-UNet”: forward() gọi thẳng SwinTransformerSys.
    Đồng nhất style với sys: logging thay print, import gọn, load_from rõ ràng hơn.
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.vis = vis

        self.swin_unet = SwinTransformerSys(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            depths_decoder=getattr(config.MODEL.SWIN, "DEPTHS_DECODER", (1, 2, 2, 2)),  # Mặc định set dim decoder = (1, 2, 2, 2)
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

    def forward(self, x):
        # Nếu ảnh xám 1 kênh màu -> lặp lại thành 3 kênh, giữ bản chất mô hình
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        logits, gating_weights = self.swin_unet(x)
        return logits, gating_weights

    @torch.no_grad()
    def load_from(self, config):
        pretrained_path = getattr(config.MODEL, "PRETRAIN_CKPT", None)
        if not pretrained_path:
            logger.info("No pretrained checkpoint provided.")
            return

        logger.info(f"Loading pretrained checkpoint from: {pretrained_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(pretrained_path, map_location=device)
        
        # Lấy state_dict từ key 'model'
        pretrained_dict = ckpt['model']
        model_dict = self.swin_unet.state_dict()
        full_dict = {}

        # 1. Xử lý lớp đầu vào patch_embed (3 kênh -> 12 kênh)
        if 'patch_embed.proj.weight' in pretrained_dict:
            target_weight = model_dict['patch_embed.proj.weight'] # Shape [96, 12, 4, 4]
            source_weight = pretrained_dict['patch_embed.proj.weight'] # Shape [96, 3, 4, 4]
            
            if source_weight.shape[1] != target_weight.shape[1]:
                logger.info(f"Mismatched input channels: {source_weight.shape[1]} -> {target_weight.shape[1]}. Repeating weights.")
                # Lặp lại trọng số 3 kênh thành 12 kênh
                repeat_times = target_weight.shape[1] // source_weight.shape[1]
                new_weight = source_weight.repeat(1, repeat_times, 1, 1)
                pretrained_dict['patch_embed.proj.weight'] = new_weight

        # 2. Map trọng số từ Encoder sang Decoder (Giữ nguyên logic Swin-Unet)
        for k, v in pretrained_dict.items():
            # Copy cho phần Encoder
            if k in model_dict and v.shape == model_dict[k].shape:
                full_dict[k] = v
            
            # Map sang phần Decoder (layers_up)
            if "layers." in k:
                # k có dạng layers.0.blocks.0... -> lấy số tầng ở index 7
                try:
                    layer_num = int(k[7:8])
                    current_layer_num = 3 - layer_num
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    if current_k in model_dict and v.shape == model_dict[current_k].shape:
                        full_dict[current_k] = v
                except:
                    continue

        # 3. Loại bỏ các lớp không khớp (như head output cuối cùng)
        for k in list(full_dict.keys()):
            if "head" in k or "output" in k:
                del full_dict[k]

        msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        logger.info(f"Successfully loaded weights with message: {msg}")