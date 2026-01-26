import torch
import torch.nn as nn
from thop import profile

from src.AsymFormer import B0_T
from src.new_asymformer import New_Asymformer, New_Asymformer_v2


if __name__ == "__main__":
    # model = B0_T(num_classes=40)
    model = New_Asymformer_v2("S", "b0", None, None, 0.7, num_classes=40)
    model.eval()
    model.cuda()
    input_rgb = torch.rand(1, 3, 480, 640).cuda()
    input_d = torch.rand(1, 1, 480, 640).cuda()
    flops, params = profile(model, inputs=(input_rgb, input_d))
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))
