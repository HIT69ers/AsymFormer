import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def generate_boundary_gt(gtmasks):
    laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
    
    fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0

    boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, stride=2, padding=1)
    boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
    
    boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, stride=4, padding=1)
    boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

    boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), laplacian_kernel, stride=8, padding=1)
    boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

    boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
    boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
    boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
    
    boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
    boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
    
    
    boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
    boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
    
    
    boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
    boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
    
    boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
    
    boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
    boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, fuse_kernel)

    boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
    boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
    return boudary_targets_pyramid


if __name__ == "__main__":
    gt = cv2.imread(r'D:/Datasets/data/labels/0.png', cv2.IMREAD_UNCHANGED)
    gt = torch.from_numpy(gt).unsqueeze(0)
    mask = generate_boundary_gt(gt).detach().cpu().numpy()
    mask = mask.squeeze()
    print(mask.shape)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
