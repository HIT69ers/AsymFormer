import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.convnext import *
from src.mix_transformer import *
from src.MLPDecoder import DecoderHead

from src.AsymFormer import Cross_Atten_Lite_split, SpatialAttention_max, SCC_Module


def get_convnext_backbone(shape='T', pretrained="/mnt/syh/pretrained/ConvNeXtV2/ImageNet-1K/convnextv2_tiny_1k_224_ema.pt", drop_path_rate=0.3):
    backbone = dict(
        T=convnext_tiny_local,
        S=convnext_small_local,
    )
    assert shape in backbone.keys()
    model = backbone[shape](pretrained=pretrained, in_22k=False, drop_path_rate=drop_path_rate)
    model_dict = dict(
        downsample_layers=[model.downsample_layers[i] for i in range(4)],
        stages=[model.stages[i] for i in range(4)]
    )
    return model_dict
