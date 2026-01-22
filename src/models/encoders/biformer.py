import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mix_transformer import *
from src.MLPDecoder import DecoderHead

from src.AsymFormer import Cross_Atten_Lite_split, SpatialAttention_max, SCC_Module
from src.myformer import get_segformer_backbone


rgb_backbone = get_segformer_backbone(shape='b2', pretrained="/home/sunyuhang/WorkSpace/pretrained/Segformer/mit_b2.pth")
d_backbone = get_segformer_backbone(shape="b0", pretrained="/home/sunyuhang/WorkSpace/pretrained/Segformer/mit_b0.pth")


class biformer_stage(nn.Module):
    def __init__(self, inc_depth, inc_rgb, stage_num, rgb_backbone, d_backbone, cut_first=False, downsample_ratio=1.0, ):
        super(biformer_stage, self).__init__()
        self.stage_num = stage_num
        self.downsample_ratio = downsample_ratio

        if cut_first:  # 设置为True时，depth分支第一个patchEmbed为单通道输入，无法加载预训练权重
            raise NotImplementedError
        else:
            self.rgb_patch_embed = rgb_backbone['patch_embeds'][self.stage_num]
            self.d_patch_embed = d_backbone['patch_embeds'][self.stage_num]
        
        self.rgb_block = rgb_backbone['blocks'][self.stage_num]
        self.d_block = d_backbone['blocks'][self.stage_num]
        self.rgb_norm = rgb_backbone['norms'][self.stage_num]
        self.d_norm = d_backbone['norms'][self.stage_num]

        if self.stage_num != 0:
            self.SCC = SCC_Module(inc_depth, inc_rgb)

    def forward(self, x_rgb: torch.Tensor, x_d: torch.Tensor):
        B, _, H, W = x_rgb.shape

        if self.stage_num == 0:
            h, w = int(H * self.downsample_ratio), int(W * self.downsample_ratio)
            x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=False)

        x_rgb, h, w = self.rgb_patch_embed(x_rgb)
        for i, blk in enumerate(self.rgb_block):
            x_rgb = blk(x_rgb, h, w)
        x_rgb = self.rgb_norm(x_rgb)
        x_rgb = x_rgb.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()

        x_d, H, W = self.d_patch_embed(x_d)
        for i, blk in enumerate(self.d_block):
            x_d = blk(x_d, H, W)
        x_d = self.d_norm(x_d)
        x_d = x_d.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # feature fusion with SCC
        if self.stage_num != 0:
            x_rgb = F.interpolate(x_rgb, (H, W), mode='bilinear', align_corners=False)
            x_fused = self.SCC(x_d, x_rgb)
            x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=False)
            return x_rgb, x_fused
        else:
            return x_rgb, x_d
        

class biformer(nn.Module):
    def __init__(self, rgb_backbone, d_backbone, num_classes=40, downsample_ratio=1.0):
        super(biformer, self).__init__()

        self.rgb_channels = [64, 128, 320, 512]
        self.d_channels = [32, 64, 160, 256]

        self.stage1 = biformer_stage(inc_depth=self.d_channels[0],
                                     inc_rgb=self.rgb_channels[0],
                                     stage_num=0,
                                     rgb_backbone=rgb_backbone,
                                     d_backbone=d_backbone,
                                     downsample_ratio=downsample_ratio)
        
        self.stage2 = biformer_stage(inc_depth=self.d_channels[1],
                                     inc_rgb=self.rgb_channels[1],
                                     stage_num=1,
                                     rgb_backbone=rgb_backbone,
                                     d_backbone=d_backbone,
                                     downsample_ratio=downsample_ratio)
        
        self.stage3 = biformer_stage(inc_depth=self.d_channels[2],
                                     inc_rgb=self.rgb_channels[2],
                                     stage_num=2,
                                     rgb_backbone=rgb_backbone,
                                     d_backbone=d_backbone,
                                     downsample_ratio=downsample_ratio)
        
        self.stage4 = biformer_stage(inc_depth=self.d_channels[3],
                                     inc_rgb=self.rgb_channels[3],
                                     stage_num=3,
                                     rgb_backbone=rgb_backbone,
                                     d_backbone=d_backbone,
                                     downsample_ratio=downsample_ratio)
        
        self.decoder = DecoderHead(in_channels=self.d_channels,
                                   num_classes=num_classes,
                                   dropout_ratio=0.1,
                                   norm_layer=nn.BatchNorm2d,
                                   embed_dim=256)
    
    def forward(self, x_rgb: torch.Tensor, x_d: torch.Tensor):
        input_shape = x_rgb.shape[-2:]
        rgb_out, d_out1 = self.stage1(x_rgb, x_d)
        rgb_out, d_out2 = self.stage2(rgb_out, d_out1)
        rgb_out, d_out3 = self.stage3(rgb_out, d_out2)
        _, d_out4 = self.stage4(rgb_out, d_out3)

        rgb_out = self.decoder([d_out1, d_out2, d_out3, d_out4])
        rgb_out = F.interpolate(rgb_out, size=input_shape, mode='bilinear', align_corners=False)
        return rgb_out