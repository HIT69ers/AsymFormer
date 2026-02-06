import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

from src.new_asymformer import New_Asymformer_v3


DOWNSAMPLE_RATIO = 0.7
MODEL_CONFIG = dict(name="new_former", 
                    rgb_branch="S", 
                    rgb_pretrained=None,
                    d_branch="b0",
                    d_pretrained=None,
                    version='v2')

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

    input_rgb = torch.randn(1, 3, 480, 640).cuda()

    input_depth = torch.randn(1, 1, 480, 640).cuda()
    # input_depth = torch.randn(1, 3, 480, 640).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input_rgb, input_depth)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input_rgb, input_depth)
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
            model(input_rgb, input_depth)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(round(FPS, 2))

    str_time = time.strftime(f"%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Current time: {str_time}")
