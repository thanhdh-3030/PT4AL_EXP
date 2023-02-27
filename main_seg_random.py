# cold start ex
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from packaging import version
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random

from models import *
from utils.utils import progress_bar
# from loader import Loader, Loader2
# from torchvision.datasets import Cityscapes
from dataloaders.datasets.cityscapes import CityscapesSegmentation
# from models.deeplabv1 import DeepLabV1
# from utils.utils import loss_calc
from utils.calculate_weights import calculate_weigths_labels
import wandb
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from models.loss import SegmentationLosses

from mmseg.models.builder import build_segmentor
norm_cfg = dict(type='BN', requires_grad=True)
model_cfg = dict(
    type='SunSegmentor',
    
    backbone=dict(
        type='ResNetV1c',
        pretrained='open-mmlab://resnet18_v1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=1024,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=True,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
    # test_cfg=None
    )
random.seed(0)
torch.manual_seed(0)

# WANDB_KEY = "f9417e07dcb47e40d732334c9390cb2929834f96"
# WANDB_HOST = "https://api.wandb.ai"
# os.environ["WANDB_API_KEY"] = WANDB_KEY
# os.environ["WANDB_BASE_URL"] = WANDB_HOST
experiment_name='psp r18 v1 weight classed no crop'
wandb.init(project="Active Learning", name=experiment_name)

# wandb.login()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
CITYSCAPES_DIR='/home/dang.hong.thanh/datasets/Cityspaces'
LEARNING_RATE = 2.5*1e-3
MOMENTUM=0.9
EPOCH=200
# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# image_transform=transforms.Compose([
#     # you can add other transformations in this list
#     transforms.ToTensor()])
# target_transform = transforms.Compose([
#     transforms.PILToTensor()
# ])
# train_dataset= Cityscapes(root=cityspace_dir, split="train",mode="fine",target_type='semantic',transform=seg_transform,target_transform=seg_transform)
# val_dataset= Cityscapes(root=CITYSCAPES_DIR, split="val",mode="fine",target_type='semantic',transform=image_transform,target_transform=target_transform)
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.resize = 512
args.base_size = 512
# args.crop_size = 321
args.crop_size = 512
args.batch_size=16
args.multi_scale= None
args.resume=False
args.cuda=True
args.loss_type='ce'
args.use_balanced_weights=True
val_dataset=CityscapesSegmentation(args,root=CITYSCAPES_DIR,split='val')
testloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,drop_last=False)

indices = list(range(2975))
random.shuffle(indices)
labeled_set = indices[:600]
# trainset=Cityscapes(root=CITYSCAPES_DIR, split="train",mode="fine",target_type='semantic',transform=image_transform,target_transform=target_transform)
train_dataset=CityscapesSegmentation(args,root=CITYSCAPES_DIR,split='train')
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=2, sampler=SubsetRandomSampler(labeled_set))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)

# trainset = Loader(is_train=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2, sampler=SubsetRandomSampler(labeled_set))

# testset = Loader(is_train=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = ResNet18()
# net = net.to(device)
if args.use_balanced_weights:
    class_weights_path=os.path.join(CITYSCAPES_DIR,'Cityscapes_classes_weights_all.npy')
    if os.path.isfile(class_weights_path):
        weight=np.load(class_weights_path)
    else:
        weight=calculate_weigths_labels(trainloader,num_classes=19)
    weight = torch.from_numpy(weight.astype(np.float32))
net = build_segmentor(model_cfg)
net.init_weights()
net = net.to(device)
# net=DeepLabV1(n_classes=19, n_blocks=[3, 4, 23, 3])
# net = net.to(device)
if version.parse(torch.__version__) >= version.parse('0.4.0'):
    # interp = nn.Upsample(size=(args.base_size, args.base_size*2), mode='bilinear', align_corners=True)
    train_interp = nn.Upsample(size=(args.crop_size, args.crop_size), mode='bilinear', align_corners=True)
    val_interp = nn.Upsample(size=(args.crop_size, args.crop_size*2), mode='bilinear', align_corners=True)

else:
    # interp = nn.Upsample(size=(args.base_size, args.base_size*2), mode='bilinear')
    interp = nn.Upsample(size=(args.crop_size, args.crop_size), mode='bilinear')

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
# optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE,momentum=MOMENTUM, weight_decay=5e-4)
optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
# criterion = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])
scheduler=LR_Scheduler(mode='cos',base_lr=LEARNING_RATE,num_epochs=EPOCH,iters_per_epoch=len(train_dataset)/2,min_lr=0.001)
criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_evaluator.reset()
    for batch_idx, samples in enumerate(trainloader):
        inputs,targets=samples['image'],samples['label']
        inputs, targets = inputs.to(device), targets.to(device)

        maxs,_=torch.topk(torch.unique(targets.flatten()),k=2)
        print(maxs[1].min())
        scheduler(optimizer, batch_idx, epoch, best_acc)
        optimizer.zero_grad()
        outputs = net(inputs)
        # outputs=F.normalize(outputs,dim=1)
        # outputs=F.softmax(outputs,dim=1)
        outputs=val_interp(outputs[0])
        # print(targets.min(),targets.max())
        # loss = loss_calc(outputs, targets,gpu='cuda:0')
        loss=criterion(outputs,targets)
        loss.backward()
        # scaler.scale(loss).backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        # total += targets.size(0)
        total +=targets.size(0)*targets.size(-1)*targets.size(-2)
        # squeeze on class channel
        targets=targets.squeeze(1)
        correct += predicted.eq(targets).sum().item()
        # Normalize correct
        # correct=correct/(targets.size(-1)*targets.size(-2))
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # if batch_idx==2: break
                    # Add batch sample into evaluator
        preds = torch.argmax(outputs, axis=1)
        train_evaluator.add_batch(targets.cpu().numpy(),preds.data.cpu().numpy())

    Acc = train_evaluator.Pixel_Accuracy()
    Acc_class = train_evaluator.Pixel_Accuracy_Class()
    mIoU = train_evaluator.Mean_Intersection_over_Union()
    FWIoU = train_evaluator.Frequency_Weighted_Intersection_over_Union()   
    wandb.log({'train/loss': train_loss/len(trainloader),
                'train/acc':correct/total,
                'train/mIOU':mIoU,
                'train/Acc':Acc,
                'train/Acc_class':Acc_class,
                'train/fwIoU':FWIoU,
                'train/lr':optimizer.param_groups[0]['lr']},step=epoch)
def valid(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    val_evaluator.reset()

    with torch.no_grad():
        for batch_idx, samples in enumerate(testloader):
            inputs,targets=samples['image'],samples['label']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs=val_interp(outputs)
        # print(targets.min(),targets.max())
            loss=criterion(outputs,targets)
            # loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            # total += targets.size(0)
            total +=targets.size(0)*targets.size(-1)*targets.size(-2)
            targets=targets.squeeze(1)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            # Add batch sample into evaluator
            preds = torch.argmax(outputs, axis=1)
            val_evaluator.add_batch(targets.cpu().numpy(),preds.data.cpu().numpy())

    # Fast test during the training
    Acc = val_evaluator.Pixel_Accuracy()
    Acc_class = val_evaluator.Pixel_Accuracy_Class()
    mIoU = val_evaluator.Mean_Intersection_over_Union()
    FWIoU = val_evaluator.Frequency_Weighted_Intersection_over_Union()
    wandb.log({'val/loss': test_loss/len(testloader),
                'val/acc':correct/total,
                'val/mIOU':mIoU,
                'val/Acc':Acc,
                'val/Acc_class':Acc_class,
                'val/fwIoU':FWIoU},step=epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer':optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

        
if __name__ == '__main__':
    # scaler = torch.cuda.amp.GradScaler()
    val_evaluator = Evaluator(num_class=19)
    train_evaluator=Evaluator(num_class=19)
    for epoch in range(start_epoch, start_epoch+EPOCH):
        train(epoch)
        valid(epoch)
        # scheduler.step()
