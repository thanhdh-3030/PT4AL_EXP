# cold start ex
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from packaging import version

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random

from models import *
from utils.utils import progress_bar
from loader import Loader, Loader2
from torchvision.datasets import Cityscapes
from models.deeplabv1 import DeepLabV1
from utils.utils import loss_calc
import wandb
random.seed(0)
torch.manual_seed(0)

# WANDB_KEY = "f9417e07dcb47e40d732334c9390cb2929834f96"
# WANDB_HOST = "https://api.wandb.ai"
# os.environ["WANDB_API_KEY"] = WANDB_KEY
# os.environ["WANDB_BASE_URL"] = WANDB_HOST
experiment_name='DeepLab v1 with 600 random sample'
# wandb.init(project="Active Learning", name=experiment_name)

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
LEARNING_RATE = 2.5e-3
MOMENTUM=0.9
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
image_transform=transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()])
target_transform = transforms.Compose([
    transforms.PILToTensor()
])
# train_dataset= Cityscapes(root=cityspace_dir, split="train",mode="fine",target_type='semantic',transform=seg_transform,target_transform=seg_transform)
val_dataset= Cityscapes(root=CITYSCAPES_DIR, split="val",mode="fine",target_type='semantic',transform=image_transform,target_transform=target_transform)
testloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2,drop_last=False)

indices = list(range(2975))
random.shuffle(indices)
labeled_set = indices[:600]
trainset=Cityscapes(root=CITYSCAPES_DIR, split="train",mode="fine",target_type='semantic',transform=image_transform,target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=2, sampler=SubsetRandomSampler(labeled_set))

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
net=DeepLabV1(n_classes=34, n_blocks=[3, 4, 23, 3])
net = net.to(device)
if version.parse(torch.__version__) >= version.parse('0.4.0'):
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
else:
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_seg/ckpt_seg_83.06129961013794.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE,momentum=MOMENTUM, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets,_) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs=F.normalize(outputs,dim=1)
        outputs=F.softmax(outputs,dim=1)
        outputs=interp(outputs)
        # print(targets.min(),targets.max())
        loss = loss_calc(outputs, targets,gpu='cuda:0')
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
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
    # wandb.log({'train/loss': train_loss/len(trainloader),'train/acc':correct/total})
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs=interp(outputs)
        # print(targets.min(),targets.max())
            loss = loss_calc(outputs, targets,gpu='cuda:0')
            # loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            # total += targets.size(0)
            total +=targets.size(0)*targets.size(-1)*targets.size(-2)
            targets=targets.squeeze(1)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # wandb.log({'val/loss': test_loss/len(testloader),'val/acc':correct/total})
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

        
if __name__ == '__main__':
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
