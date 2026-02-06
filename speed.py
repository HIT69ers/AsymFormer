import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from thop import profile

from src.new_asymformer import New_Asymformer_v3


DOWNSAMPLE_RATIO = 1.0
MODEL_CONFIG = dict(name="new_former", 
                    rgb_branch="S", 
                    rgb_pretrained=None,
                    d_branch="b0",
                    d_pretrained=None,
                    version='v3')

network = New_Asymformer_v3


if __name__ == '__main__':


    import time
    device = torch.device('cuda')
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    model = network(rgb_branch=MODEL_CONFIG['rgb_branch'],
                            rgb_pretrained=MODEL_CONFIG['rgb_pretrained'],
                            d_branch=MODEL_CONFIG['d_branch'],
                            d_pretrained=MODEL_CONFIG['d_pretrained'],
                            downsample_ratio=DOWNSAMPLE_RATIO,
                            num_classes=40)
    model.eval()
    model.to(device)
    iterations = None

    input = (torch.randn(1, 3, 480, 640).cuda(), torch.randn(1, 1, 480, 640).cuda())
    # input = (torch.randn(1, 3, 480, 640).cuda(), torch.randn(1, 3, 480, 640).cuda())

    with torch.no_grad():
        for _ in range(10):
            model(*input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(*input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(*input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(round(FPS, 2))

    flops, params = profile(model, inputs=input)
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))