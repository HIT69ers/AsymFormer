import torch
from utils.utils import intersectionAndUnion, AverageMeter


def val(model, dataloader, engine):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, (sample, label) in enumerate(dataloader):
            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                if ((batch_idx + 1) % int(len(dataloader) * 0.5) == 0 or batch_idx == 0):
                    print(f"Validation Iter: {batch_idx + 1} / {len(dataloader)}")
            image = sample['img'].cuda()
            depth = sample['depth'].cuda()
            label = label.numpy()  # shape: (B, H, W) or (H, W)

            pred = model(image, depth)
            output = torch.max(pred, 1)[1].cpu().numpy()  # shape: (B, H, W) or (H, W)

            # 原代码使用了 +1，这里保留以保持与原评估脚本一致。
            # 如果你的标签是 0..C-1，请移除下面这一行或调整为与标签一致。
            # output = output + 1

            # 处理 batch 维度：若为批量（3D），逐样本计算 intersection/union
            if output.ndim == 3:
                for i in range(output.shape[0]):
                    out_i = output[i]
                    lab_i = label[i]
                    intersection, union = intersectionAndUnion(out_i, lab_i, numClass=25)
                    intersection_meter.update(intersection)
                    union_meter.update(union)
            else:
                intersection, union = intersectionAndUnion(output, label, numClass=25)
                intersection_meter.update(intersection)
                union_meter.update(union)
    
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean()
    return round(miou*100, 2)