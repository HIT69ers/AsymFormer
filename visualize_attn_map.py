import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

from src.AsymFormer import B0_T, SpatialAttention_max


def _load_block_pretrain_weight(model, pretrain_path):
    model_dict = model.state_dict()
    pretrain_dict = torch.load(pretrain_path)['state_dict']
    new_state_dict = OrderedDict()
    new_state_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}

    model.load_state_dict(new_state_dict)


def visualize_attention_on_image(img_path, depth_path, model_weight_path, save_path='result.png'):
    # 1. 硬件设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载模型
    model = B0_T(num_classes=40) # 根据你的训练配置修改 num_classes
    _load_block_pretrain_weight(model, model_weight_path)
    model.to(device)
    model.eval()

    # 3. 注册 Hook 来捕获 SpatialAttention_max 的输出
    # 我们创建一个列表来存储捕获到的 map
    attention_maps = []

    def hook_fn(module, input, output):
        # 根据代码，SpatialAttention_max 的返回值是 map * x * y_channel
        # 但我们需要的是那个中间变量 'map'
        # 注意：原代码中 forward 最后一行返回的是复合结果。
        # 为了拿到的只是 map，我们需要稍微变通一下，或者 Hook 住计算 map 的那一步。
        # 由于 'map' 是局部变量，Hook 只能拿到整个函数的 return。
        # 这里的 output 是 (map * x * y_channel)
        # 我们对 output 在通道维度取平均或最大值来近似还原空间注意力分布
        attention_maps.append(output.detach().cpu())

    # 找到所有的 SpatialAttention_max 模块并注册钩子
    for name, module in model.named_modules():
        if isinstance(module, SpatialAttention_max):
            print(f"Registering hook for: {name}")
            module.register_forward_hook(hook_fn)

    # 4. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        # 如果训练时用了 Normalize，请在此处加上
        # transforms.Normalize(mean=[...], std=[...])
    ])

    # 读取 RGB 和 Depth
    raw_img = Image.open(img_path).convert('RGB')
    raw_depth = Image.open(depth_path).convert('L') # 假设深度图是单通道

    img_tensor = transform(raw_img).unsqueeze(0).to(device)
    depth_tensor = transforms.ToTensor()(transforms.Resize((480, 640))(raw_depth)).unsqueeze(0).to(device)

    # 5. 前向传播
    with torch.no_grad():
        _ = model(img_tensor, depth_tensor)

    # 6. 处理和叠加 Attention Map
    # 注意：B0_T 中有 3 个 SCC 模块，所以会有 3 个 attention_maps
    # 我们取最后一个（通常语义最丰富）或者循环输出
    for i, attn in enumerate(attention_maps):
        # attn shape: [1, C, H, W]
        # 对通道取均值，得到 [H, W] 的热力图
        mask = torch.mean(attn, dim=1).squeeze().numpy()
        
        # 归一化到 0-255
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = np.uint8(255 * mask)

        # 调整大小回原图尺寸
        img_cv = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (640, 480))
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))

        # 叠加
        result = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
        
        # 保存
        out_name = f'attn_layer_{i}_{save_path}'
        cv2.imwrite(out_name, result)
        print(f"Saved: {out_name}")

if __name__ == '__main__':
    # 修改以下路径进行测试
    data_dir_path = '/home/syh/WorkSpace/datasets/data'
    visualize_attention_on_image(
        img_path=os.path.join(data_dir_path, 'images', '0.png'), 
        depth_path=os.path.join(data_dir_path, 'depths', '0.png'), 
        model_weight_path='/home/syh/WorkSpace/checkpoints/AsymFormer_NYUv2.pth'
    )