import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.convnextv2 import *
from src.MLPDecoder import DecoderHead

from src.AsymFormer import Cross_Atten_Lite_split, SpatialAttention_max, SCC_Module
from src.myformer import get_convnextv2_backbone


rgb_backbone = get_convnextv2_backbone()
# d_backbone = get_convnextv2_backbone(shape="A", pretrained="/home/sunyuhang/WorkSpace/pretrained/convnextv2_atto_1k_224_ema.pt", drop_path_rate=0.3)
d_backbone = get_convnextv2_backbone(shape="A", pretrained=None, drop_path_rate=0.3)


class binext_stage(nn.Module):
    def __init__(self, inc_depth, inc_rgb, stage_num, cut_first=False):
        super(binext_stage, self).__init__()
        self.stage_num = stage_num

        if cut_first:  # 设置为True时，depth分支第一个patchEmbed为单通道输入，无法加载预训练权重
            raise NotImplementedError
        else:
            self.rgb_downsample_layer = rgb_backbone['downsample_layers'][self.stage_num]
            self.d_downsample_layer = d_backbone['downsample_layers'][self.stage_num]
        
        self.rgb_stage = rgb_backbone['stages'][self.stage_num]
        self.d_stage = d_backbone['stages'][self.stage_num]

        if self.stage_num != 0:
            self.SCC = SCC_Module(inc_depth, inc_rgb)

    def forward(self, x_rgb: torch.Tensor, x_d: torch.Tensor):
        B = x_rgb.shape[0]

        x_rgb = self.rgb_downsample_layer(x_rgb)
        x_rgb = self.rgb_stage(x_rgb)

        x_d = self.d_downsample_layer(x_d)
        x_d = self.d_stage(x_d)

        # feature fusion with SCC
        if self.stage_num != 0:
            x_fused = self.SCC(x_d, x_rgb)
            return x_rgb, x_fused
        else:
            return x_rgb, x_d
        

class BiNextv2(nn.Module):
    def __init__(self, num_classes=40):
        super(BiNextv2, self).__init__()

        self.rgb_channels = [96, 192, 384, 768]
        self.d_channels = [40, 80, 160, 320]

        self.stage1 = binext_stage(inc_depth=self.d_channels[0],
                                 inc_rgb=self.rgb_channels[0],
                                 stage_num=0)
        self.stage2 = binext_stage(inc_depth=self.d_channels[1],
                                 inc_rgb=self.rgb_channels[1],
                                 stage_num=1)
        self.stage3 = binext_stage(inc_depth=self.d_channels[2],
                                 inc_rgb=self.rgb_channels[2],
                                 stage_num=2)
        self.stage4 = binext_stage(inc_depth=self.d_channels[3],
                                 inc_rgb=self.rgb_channels[3],
                                 stage_num=3)
        
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
