import torch
import torch.nn as nn
import torch.nn.functional as F

from src.spe_attn import SCC_Module
from src.convnext import convnext_tiny_local, convnext_small_local
from src.mix_transformer import get_segformer_backbone, OverlapPatchEmbed
from src.MLPDecoder import DecoderHead


class down_sample_block(nn.Module):
    def __init__(self, inc_depth, inc_rgb, block_num, rgb_backbone, d_backbone, downsample_ratio=1.0):
        super(down_sample_block, self).__init__()
        self.block_num = block_num
        self.downsample_ratio = downsample_ratio

        # rgb & d backbone
        stem1 = rgb_backbone.downsample_layers
        layers1 = rgb_backbone.stages
        stem2 = d_backbone['patch_embeds']
        layers2 = d_backbone['blocks']
        norm2 = d_backbone['norms']

        if block_num != 0:
            self.depth_stem = stem2[block_num]
            self.rgb_stem = stem1[block_num]
        else:
            self.depth_stem = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_stem = stem1[0]

        self.rgb_layer = layers1[block_num]
        self.depth_layer = layers2[block_num]

        self.depth_norm = norm2[block_num]

        if self.block_num != 0:
            self.SCC = SCC_Module(inc_depth2=inc_depth, inc_rgb=inc_rgb)

    def forward(self, image: torch.Tensor, depth: torch.Tensor):
        B, _, H, W = image.shape

        if self.block_num == 0:
            h, w = int(H * self.downsample_ratio), int(W * self.downsample_ratio)
            image = F.interpolate(image, (h, w), mode='bilinear', align_corners=False)

        image = self.rgb_stem(image)
        _, _, h, w = image.shape
        rgb_out = self.rgb_layer(image)

        depth_out, H, W = self.depth_stem(depth)

        for i, blk in enumerate(self.depth_layer):
            depth_out = blk(depth_out, H, W)
        depth_out = self.depth_norm(depth_out)
        depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # SCC_Ablation
        if self.block_num != 0:
            rgb_out = F.interpolate(rgb_out, (H, W), mode='bilinear', align_corners=False)
            merge = self.SCC(depth_out, rgb_out)
            rgb_out = F.interpolate(rgb_out, (h, w), mode='bilinear', align_corners=False)
            return rgb_out, merge
        else:
            return rgb_out, depth_out
        

class New_Asymformer(nn.Module):
    def __init__(self, rgb_branch, d_branch, rgb_pretrained, d_pretrained, downsample_ratio=1.0, num_classes=40):
        super(New_Asymformer, self).__init__()
        assert rgb_branch in ['T', 'S']
        assert d_branch in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']

        if rgb_branch == 'T':
            rgb_backbone = convnext_tiny_local(pretrained=rgb_pretrained, drop_path_rate=0.3, num_classes=1000)
            self.rgb_channels = [96, 192, 384, 768]
        elif rgb_branch == 'S':
            rgb_backbone = convnext_small_local(pretrained=rgb_pretrained, drop_path_rate=0.3, num_classes=1000)
            self.rgb_channels = [96, 192, 384, 768]
        
        d_backbone = get_segformer_backbone(shape=d_branch, pretrained=d_pretrained)
        if d_branch == 'b0':
            self.d_channels = [32, 64, 160, 256]
        else:
            self.d_channels = [64, 128, 320, 512]

        self.down_sample_1 = down_sample_block(inc_depth=self.d_channels[0], 
                                               inc_rgb=self.rgb_channels[0], 
                                               block_num=0,
                                               rgb_backbone=rgb_backbone,
                                               d_backbone=d_backbone,
                                               downsample_ratio=downsample_ratio)
        
        self.down_sample_2 = down_sample_block(inc_depth=self.d_channels[1], 
                                               inc_rgb=self.rgb_channels[1], 
                                               block_num=1,
                                               rgb_backbone=rgb_backbone,
                                               d_backbone=d_backbone,
                                               downsample_ratio=downsample_ratio)
        
        self.down_sample_3 = down_sample_block(inc_depth=self.d_channels[2], 
                                               inc_rgb=self.rgb_channels[2], 
                                               block_num=2,
                                               rgb_backbone=rgb_backbone,
                                               d_backbone=d_backbone,
                                               downsample_ratio=downsample_ratio)
        
        self.down_sample_4 = down_sample_block(inc_depth=self.d_channels[3], 
                                               inc_rgb=self.rgb_channels[3], 
                                               block_num=3,
                                               rgb_backbone=rgb_backbone,
                                               d_backbone=d_backbone,
                                               downsample_ratio=downsample_ratio)
        
        self.Decoder = DecoderHead(in_channels=self.d_channels, num_classes=num_classes, dropout_ratio=0.1,
                                   norm_layer=nn.BatchNorm2d,
                                   embed_dim=256)

    def forward(self, image, depth):
        input_shape = image.shape[-2:]

        rgb_out, depth_out1 = self.down_sample_1(image, depth)
        rgb_out, depth_out2 = self.down_sample_2(rgb_out, depth_out1)

        rgb_out, depth_out3 = self.down_sample_3(rgb_out, depth_out2)
        _, depth_out = self.down_sample_4(rgb_out, depth_out3)

        rgb_out = self.Decoder(
            [depth_out1,
             depth_out2,
             depth_out3,
             depth_out])
        rgb_out = F.interpolate(rgb_out, size=input_shape, mode='bilinear', align_corners=False)
        return rgb_out