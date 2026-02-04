import os
import cv2
import torch
import numpy as np

from src.loss.detail_loss import DetailAggregateLoss


MEMORY_PATH = "/mnt/syh"
dataset_path = os.path.join(MEMORY_PATH, "datasets", "NYUv2", "data")


if __name__ == "__main__":
    torch.manual_seed(15)
    img_path = os.path.join(dataset_path, "labels", "0.png")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    label = img.astype(np.int16)
    img_tensor = torch.from_numpy(label).cuda()
    img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor)

    detailAggregateLoss = DetailAggregateLoss()
    for param in detailAggregateLoss.parameters():
        print(param)

    bce_loss,  dice_loss = detailAggregateLoss(torch.unsqueeze(img_tensor, 0), img_tensor)
    print(bce_loss,  dice_loss)