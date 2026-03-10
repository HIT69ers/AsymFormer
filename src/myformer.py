import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.convnextv2 import *
from src.mix_transformer import *
from src.MLPDecoder import DecoderHead

from src.AsymFormer import Cross_Atten_Lite_split, SpatialAttention_max, SCC_Module

sys.path.append('..')

"""
params:
shape: {A, F, P, N, T, B, L, H}
pretrained: None or str

return:
{downsample_layers: [], stages: []}
"""
def get_convnextv2_backbone(shape='T', pretrained="/mnt/syh/pretrained/ConvNeXtV2/ImageNet-1K/convnextv2_tiny_1k_224_ema.pt", drop_path_rate=0.3):
    backbone = dict(
        A=convnextv2_atto,
        F=convnextv2_femto,
        P=convnext_pico,
        N=convnextv2_nano,
        T=convnextv2_tiny,
        B=convnextv2_nano,
        L=convnextv2_large,
        H=convnextv2_huge,
    )
    assert shape in backbone.keys()
    model = backbone[shape](drop_path_rate=drop_path_rate)
    if pretrained is not None:
        print(f"Loading pretrained checkpoint from {pretrained}")
        checkpoint = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
    model_dict = dict(
        downsample_layers=[model.downsample_layers[i] for i in range(4)],
        stages=[model.stages[i] for i in range(4)]
    )
    return model_dict


"""
params:
shape: {b0, b1, b2, b3, b4, b5}
"""
def get_segformer_backbone(shape='b0', pretrained=None):
    backbone = dict(
        b0=mit_b0,
        b1=mit_b1,
        b2=mit_b2,
        b3=mit_b3,
        b4=mit_b4,
        b5=mit_b5,
    )
    assert shape in backbone.keys()
    model = backbone[shape]()
    net_dict =  model.state_dict()

    if pretrained is not None:
        print(f"Loading pretrained checkpoint from {pretrained}")
        checkpoint = torch.load(pretrained, map_location="cpu")
        checkpoint_dict = {k: v for k, v in list(checkpoint.items()) if k in net_dict}
        net_dict.update(checkpoint_dict)
        model.load_state_dict(net_dict, strict=True)
    
    model_dict = dict(
        patch_embeds=[model.patch_embed1, model.patch_embed2, model.patch_embed3, model.patch_embed4],
        blocks=[model.block1, model.block2, model.block3, model.block4],
        norms=[model.norm1, model.norm2, model.norm3, model.norm4]
    )
    return model_dict


rgb_backbone = get_convnextv2_backbone(pretrained=None)
d_backbone = get_segformer_backbone()


class RGBD_Stage(nn.Module):
    def __init__(self, inc_depth, inc_rgb, stage_num, cut_first=True):
        super(RGBD_Stage, self).__init__()
        self.stage_num = stage_num

        if cut_first:  # 设置为True时，depth分支第一个patchEmbed为单通道输入，无法加载预训练权重
            if self.stage_num != 0:
                self.d_patch_embed = d_backbone['patch_embeds'][self.stage_num]
            else:
                self.d_patch_embed = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_downsample_layer = rgb_backbone['downsample_layers'][self.stage_num]
        else:
            raise NotImplementedError
        
        self.rgb_stage = rgb_backbone['stages'][self.stage_num]
        self.d_block = d_backbone['blocks'][self.stage_num]
        self.d_norm = d_backbone['norms'][self.stage_num]

        if self.stage_num != 0:
            self.SCC = SCC_Module(inc_depth, inc_rgb)

    def forward(self, x_rgb: torch.Tensor, x_d: torch.Tensor):
        B = x_rgb.shape[0]

        x_rgb = self.rgb_downsample_layer(x_rgb)
        x_rgb = self.rgb_stage(x_rgb)

        x_d, H, W = self.d_patch_embed(x_d)
        for i, blk in enumerate(self.d_block):
            x_d = blk(x_d, H, W)
        x_d = self.d_norm(x_d)
        x_d = x_d.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # feature fusion with SCC
        if self.stage_num != 0:
            x_fused = self.SCC(x_d, x_rgb)
            return x_rgb, x_fused
        else:
            return x_rgb, x_d


"""
convnextv2(RGB) + segformer(Depth)
"""
class MyFormer(nn.Module):
    def __init__(self, num_classes=40):
        super(MyFormer, self).__init__()

        self.rgb_channels = [96, 192, 384, 768]
        self.d_channels = [32, 64, 160, 256]

        self.stage1 = RGBD_Stage(inc_depth=self.d_channels[0],
                                 inc_rgb=self.rgb_channels[0],
                                 stage_num=0)
        self.stage2 = RGBD_Stage(inc_depth=self.d_channels[1],
                                 inc_rgb=self.rgb_channels[1],
                                 stage_num=1)
        self.stage3 = RGBD_Stage(inc_depth=self.d_channels[2],
                                 inc_rgb=self.rgb_channels[2],
                                 stage_num=2)
        self.stage4 = RGBD_Stage(inc_depth=self.d_channels[3],
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


if __name__ == "__main__":
    get_convnextv2_backbone()
