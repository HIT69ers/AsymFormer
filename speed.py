import os
import time
import random
import argparse
from importlib import import_module
import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from models.builder import EncoderDecoder as segmodel
import torch.nn as nn
from thop import profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    device = torch.device('cuda')
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    args = parse_args()
    cfg = getattr(import_module(args.config), "C")
    if os.path.exists(cfg.log_dir):
        os.rmdir(cfg.log_dir)
    print("############################################")
    print(f"Testing speed of {cfg.name}")
    print("############################################")
    criterion = nn.CrossEntropyLoss(reduction="mean", 
                                    ignore_index=cfg.background)
    BatchNorm2d = nn.SyncBatchNorm

    model = segmodel(
        cfg=cfg,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=True,
    )
    model.eval()
    model.to(device)

    dummy_input = (torch.rand(1, 3, 480, 640).cuda(), 
                   torch.rand(1, 3, 480, 640).cuda())
    # flops, params = profile(model, inputs=dummy_input)
    # print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))

    iterations = None

    with torch.no_grad():
        for _ in range(10):
            model(*dummy_input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(*dummy_input)
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
            model(*dummy_input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)

