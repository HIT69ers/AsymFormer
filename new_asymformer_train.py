'''
Our code is partially adapted from RedNet (https://github.com/JinDongJiang/RedNet)
'''
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch import nn
from src.new_asymformer import New_Asymformer, New_Asymformer_v2, New_Asymformer_v3, New_Asymformer_v3_loss, New_Asymformer_v3_dloss
import NYUv2_dataloader as Data
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
import random
import datetime

from src.loss.detail_loss import DetailAggregateLoss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

IGNORE_INDEX = -1  
DECODER_LOSS = True
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
DOWNSAMPLE_RATIO = 1.0
MEMORY_PATH = "/mnt/syh"
MODEL_CONFIG = dict(name="new_former", 
                    rgb_branch="S", 
                    rgb_pretrained=os.path.join(MEMORY_PATH, "pretrained", "convnext", "convnext_small_1k_224_ema.pth"),
                    d_branch="b0",
                    d_pretrained=None,
                    version='v3',
                    with_4=False,
                    with_8=False,
                    with_16=False,
                    with_32=True)
print("===================Train Config===================")
for k, v in MODEL_CONFIG.items():
    print(f"{k}: {v}")
print(f"Downsample_ratio: {DOWNSAMPLE_RATIO}")
print(f"Ignore_index: {IGNORE_INDEX}")
print("==================================================")

detail_str = ""
if DECODER_LOSS:
    detail_str += "_dloss"
if MODEL_CONFIG['with_4']:
    detail_str += "_4"
if MODEL_CONFIG['with_8']:
    detail_str += "_8"
if MODEL_CONFIG['with_16']:
    detail_str += "_16"
if MODEL_CONFIG['with_32']:
    detail_str += '_32'

detail_str += "_only_dice"


dataset_path = os.path.join(MEMORY_PATH, "datasets", "NYUv2", "data")
ckpt_dir = os.path.join(MEMORY_PATH, "asym_checkpoints", MODEL_CONFIG['name'] + "_" + MODEL_CONFIG['rgb_branch'] + "_" + MODEL_CONFIG['d_branch'] + '_' +\
                        str(DOWNSAMPLE_RATIO) + "_" + MODEL_CONFIG['version'] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + detail_str)

# print(f"use detail loss in the end")
# ckpt_dir += "_end"

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default=dataset_path, metavar='DIR',
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
parser.add_argument('--ckpt-dir', default=ckpt_dir, metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')


args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480


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


def train():
    setup_seed(2333)
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

    num_train = len(train_data)

    ######################################
    # Network
    if MODEL_CONFIG['version'] == 'v1':
        network = New_Asymformer
    elif MODEL_CONFIG['version'] == 'v2':
        network = New_Asymformer_v2
    elif MODEL_CONFIG['version'] == 'v3':
        if not (MODEL_CONFIG['with_4'] or MODEL_CONFIG['with_8'] or MODEL_CONFIG['with_16'] or MODEL_CONFIG['with_32']):
            network = New_Asymformer_v3
        else:
            print(f"Using detail loss")
            if not DECODER_LOSS:
                network = New_Asymformer_v3_loss
            else:
                print(f"Decoder detail loss")
                network = New_Asymformer_v3_dloss
        
    model = network(rgb_branch=MODEL_CONFIG['rgb_branch'],
                    rgb_pretrained=MODEL_CONFIG['rgb_pretrained'],
                    d_branch=MODEL_CONFIG['d_branch'],
                    d_pretrained=MODEL_CONFIG['d_pretrained'],
                    downsample_ratio=DOWNSAMPLE_RATIO,
                    num_classes=40,
                    with_4=MODEL_CONFIG['with_4'],
                    with_8=MODEL_CONFIG['with_8'],
                    with_16=MODEL_CONFIG['with_16'],
                    with_32=MODEL_CONFIG['with_32'])
    #####################################

    ##################################### 
    # Loss
    CEL_weighted = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    detail_loss = DetailAggregateLoss()
    #####################################

    model.train()
    model.to(device)
    CEL_weighted.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.weight_decay)
    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    for epoch in range(int(args.start_epoch), args.epochs):

        local_count = 0
        last_count = 0
        end_time = time.time()
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                      local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label']]

            optimizer.zero_grad()

            # Inference
            if (not MODEL_CONFIG['with_4']) and (not MODEL_CONFIG['with_8']) and (not MODEL_CONFIG['with_16']) and (not MODEL_CONFIG['with_32']):
                out = model(image, depth)
            if (not MODEL_CONFIG['with_4']) and (not MODEL_CONFIG['with_8']) and (not MODEL_CONFIG['with_16']) and MODEL_CONFIG['with_32']:
                out, out32 = model(image, depth)
            if (not MODEL_CONFIG['with_4']) and (not MODEL_CONFIG['with_8']) and MODEL_CONFIG['with_16'] and MODEL_CONFIG['with_32']:
                out, out16, out32 = model(image, depth)
            if (not MODEL_CONFIG['with_4']) and MODEL_CONFIG['with_8'] and MODEL_CONFIG['with_16'] and MODEL_CONFIG['with_32']:
                out, out8, out16, out32 = model(image, depth)
            if MODEL_CONFIG['with_4'] and MODEL_CONFIG['with_8'] and MODEL_CONFIG['with_16'] and MODEL_CONFIG['with_32']:
                out, out4, out8, out16, out32 = model(image, depth)
            if MODEL_CONFIG['with_4'] and (not MODEL_CONFIG['with_8']) and (not MODEL_CONFIG['with_16']) and (not MODEL_CONFIG['with_32']):
                out, out4 = model(image, depth)
            
            # calculate loss
            loss = CEL_weighted(out, (target_scales[0] - 1).long())

            boundery_bce_loss = 0.
            boundery_dice_loss = 0.

            # if 'end' in ckpt_dir:
            #     boundery_bce_loss, boundery_dice_loss = detail_loss(out, target_scales[0].long())

            if MODEL_CONFIG['with_4']:
                boundery_bce_loss4, boundery_dice_loss4 = detail_loss(out4, target_scales[0])
                boundery_bce_loss += boundery_bce_loss4
                boundery_dice_loss += boundery_dice_loss4
            if MODEL_CONFIG['with_8']:
                boundery_bce_loss8, boundery_dice_loss8 = detail_loss(out8, target_scales[0])
                boundery_bce_loss += boundery_bce_loss8
                boundery_dice_loss += boundery_dice_loss8
            if MODEL_CONFIG['with_16']:
                boundery_bce_loss16, boundery_dice_loss16 = detail_loss(out16, target_scales[0])
                boundery_bce_loss += boundery_bce_loss16
                boundery_dice_loss += boundery_dice_loss16
            if MODEL_CONFIG['with_32']:
                boundery_bce_loss32, boundery_dice_loss32 = detail_loss(out32, target_scales[0])
                boundery_bce_loss += boundery_bce_loss32
                boundery_dice_loss += boundery_dice_loss32

            # bce + dice
            # loss += boundery_bce_loss + boundery_dice_loss
            # only bce
            # loss += boundery_bce_loss
            # only dice
            loss += boundery_dice_loss

            # iteration
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            local_count += image.data.shape[0]
            global_step += 1

            if global_step % args.print_freq == 0 or global_step == 1:
                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()
                last_count = local_count

    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
              0, num_train)

    print("Training completed ")


if __name__ == '__main__':
    try:
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)

        train()
    except:
        os.remove(args.ckpt_dir)
