'''
Our code is partially adapted from RedNet (https://github.com/JinDongJiang/RedNet)
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch import nn
from src.B0_S import B0_S
import NYUv2_dataloader as Data
from utils.utils import save_ckpt, save_ckpt_new, intersectionAndUnion
from utils.utils import load_ckpt, AverageMeter
from utils.utils import print_log
from utils.logger import get_logger
import random

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default="/mnt/sdb/syh/datasets/data/", metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.01, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=25, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt-dir', default='/mnt/sdb/syh/asym_checkpoints/B0_S_0.6_auto_bsize8', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')
parser.add_argument('--amp', action='store_true', default=False,
                    help="autocast train")

DOWNSAMPLE_RATIO = 0.6

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480


def is_eval(epoch):
    return epoch > 250 or epoch == 1 or epoch % 10 == 0


class Engine(object):
    def __init__(self):
        self.checkpoint_state = []
        self.logger = get_logger()

    def save_and_remove(self, epoch, miou, ckpt_dir, model, optimizer, global_step):
        self.checkpoint_state.append(dict(epoch=epoch, miou=miou))
        self.checkpoint_state.sort(key=lambda x: x["miou"], reverse=True)
        if len(self.checkpoint_state) > 5:
            try:
                ckpt_model_filename = f"epoch-{self.checkpoint_state[-1]['epoch']}_miou-{self.checkpoint_state[-1]['miou']}.pth"
                ckpt_path = os.path.join(ckpt_dir, ckpt_model_filename)
                os.remove(ckpt_path)
                self.logger.info(f"remove inferior checkpoint: {self.checkpoint_state[-1]}")
            except:
                pass
            self.checkpoint_state.pop()
        save_ckpt_new(ckpt_dir, model, optimizer, global_step, epoch, 0, 1, miou)
        


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=4,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def val(model, dataloader, device):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            if ((batch_idx + 1) % int(len(dataloader) * 0.5) == 0 or batch_idx == 0):
                print(f"Validation Iter: {batch_idx + 1} / {len(dataloader)}")
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].numpy()  # shape: (B, H, W) or (H, W)

            pred = model(image, depth)
            output = torch.max(pred, 1)[1].cpu().numpy()  # shape: (B, H, W) or (H, W)

            # 原代码使用了 +1，这里保留以保持与原评估脚本一致。
            # 如果你的标签是 0..C-1，请移除下面这一行或调整为与标签一致。
            output = output + 1

            # 处理 batch 维度：若为批量（3D），逐样本计算 intersection/union
            if output.ndim == 3:
                for i in range(output.shape[0]):
                    out_i = output[i]
                    lab_i = label[i]
                    intersection, union = intersectionAndUnion(out_i, lab_i, numClass=40)
                    intersection_meter.update(intersection)
                    union_meter.update(union)
            else:
                intersection, union = intersectionAndUnion(output, label, numClass=40)
                intersection_meter.update(intersection)
                union_meter.update(union)
    
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean()
    return round(miou*100, 2)


def train():

    logger = get_logger(log_dir=os.path.join(args.ckpt_dir, "log"),
                        log_file="train.log")
    
    engine = Engine()

    best_miou = 0
    
    seed = 2333
    setup_seed(seed)
    logger.info(f"set seed {seed}")

    train_data = Data.RGBD_Dataset(transform=transforms.Compose([Data.scaleNorm(),
                                                                 Data.RandomScale((1.0, 1.4, 2.0)),
                                                                 Data.RandomHSV((0.9, 1.1),
                                                                                (0.9, 1.1),
                                                                                (25, 25)),
                                                                 Data.RandomCrop(image_h, image_w),
                                                                 Data.RandomFlip(),
                                                                 Data.ToTensor(),
                                                                 Data.Normalize()]),
                                   phase_train=True,
                                   data_dir=args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    
    val_data = Data.RGBD_Dataset(transform=transforms.Compose([Data.scaleNorm(),
                                                               Data.ToTensor(),
                                                               Data.Normalize()]),
                                 phase_train=False,
                                 data_dir=args.data_dir,
                                 txt_name='test.txt'
                                 )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_train = len(train_data)

    model = B0_S(num_classes=40, downsample_ratio=DOWNSAMPLE_RATIO)

    CEL_weighted = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

    model.train()
    model.to(device)
    CEL_weighted.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.weight_decay)
    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(int(args.start_epoch), args.epochs):
        model.train()
        local_count = 0
        last_count = 0
        end_time = time.time()
        # if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
        #     save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
        #               local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label']]

            optimizer.zero_grad()

            if args.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(image, depth)
                    loss = CEL_weighted(out, (target_scales[0] - 1).long())
            else:
                out = model(image, depth)
                loss = CEL_weighted(out, (target_scales[0] - 1).long())

            if args.amp:
                # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            local_count += image.data.shape[0]
            global_step += 1
            real_epoch = epoch + 1

            if global_step % args.print_freq == 0 or global_step == 1:
                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, real_epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()
                last_count = local_count

        if is_eval(real_epoch):
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                if args.amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        miou = val(model, val_loader, device)
                else:
                    miou = val(model, val_loader, device)
            if miou > best_miou:
                best_miou = miou
                engine.save_and_remove(real_epoch, miou, args.ckpt_dir, model, optimizer, global_step)
            
            logger.info(f"Epoch {real_epoch} validation result: mIoU {miou}, best mIoU {best_miou}")

    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
              0, num_train)

    print("Training completed ")


if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    train()
