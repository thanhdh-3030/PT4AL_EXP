'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pathlib import Path
from packaging import version

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np
from PIL import Image

from models import *
from loader import Loader, Loader2
from utils.utils import progress_bar
from torchvision.datasets import Cityscapes
from models.deeplabv1 import DeepLabV1
from utils.utils import loss_calc
# train_dir_img = Path('./DATA/train/')
# train_dir_mask = Path('./DATA/train/')
# val_dir_img=Path('./DATA/test')
# val_dir_mask=Path('./DATA/test')
from active_cityscape import ActiveCityscapes
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
cityspace_dir='/home/dang.hong.thanh/datasets/Cityspaces'
parser = argparse.ArgumentParser(description='PyTorch Cityscapes Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
LEARNING_RATE = 2.5e-3
MOMENTUM=0.9
# def resize_labels(labels, size):
#     """
#     Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
#     Other nearest methods result in misaligned labels.
#     -> F.interpolate(labels, shape, mode='nearest')
#     -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
#     """
#     new_labels = []
#     for label in labels:
#         # label = label.float().numpy()
#         label = label.cpu().float().numpy().squeeze(0)
#         label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
#         new_labels.append(np.asarray(label))
#     new_labels = torch.LongTensor(new_labels)
#     return new_labels
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# # testset = Loader(is_train=False, transform=transform_test)
# testset = Loader2(is_train=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')
image_transform=transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()])
target_transform = transforms.Compose([
    transforms.PILToTensor()
])
# train_dataset= Cityscapes(root=cityspace_dir, split="train",mode="fine",target_type='semantic',transform=seg_transform,target_transform=seg_transform)
val_dataset= Cityscapes(root=cityspace_dir, split="val",mode="fine",target_type='semantic',transform=image_transform,target_transform=target_transform)

# Model
print('==> Building model..')
# net = ResNet18()
net=DeepLabV1(n_classes=34, n_blocks=[3, 4, 23, 3])
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
if version.parse(torch.__version__) >= version.parse('0.4.0'):
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
else:
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
# Training
testloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2,drop_last=False)
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

def test(net, criterion, epoch, cycle):
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_seg'):
            os.mkdir('checkpoint_seg')
        torch.save(state, f'./checkpoint_seg/main_{cycle}.pth')
        best_acc = acc

# class-balanced sampling (pseudo labeling)
def get_plabels(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(10)]

    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    # overflow goes into remaining
    remaining = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if len(class_dict[predicted.item()]) < 100:
                class_dict[predicted.item()].append(samples[idx])
            else:
                remaining.append(samples[idx])
            progress_bar(idx, len(ploader))

    sample1k = []
    for items in class_dict.values():
        if len(items) == 100:
            sample1k.extend(items)
        else:
            # supplement samples from remaining 
            sample1k.extend(items)
            add = 100 - len(items)
            sample1k.extend(remaining[:add])
            remaining = remaining[add:]
    
    return sample1k

# confidence sampling (pseudo labeling)
# return 1k samples w/ lowest top1 score
def get_plabels2(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(10)]

    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores, predicted = outputs.max(1)
            # save top1 confidence score 
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            # top1_scores.append(probs[0][predicted.item()])
            top1_scores.append(probs[0][predicted.item()].cpu().item())
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:1000]]

#
def get_seg_plabels_confidence(net, samples, cycle):
    # dictionary with 10 keys as class labels
    # class_dict = {}
    # [class_dict.setdefault(x,[]) for x in range(10)]

    sample1k = []
    subset = ActiveCityscapes(cityspace_dir,transform=image_transform,target_transform=target_transform,image_files=samples)
    ploader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)

    uncertainty_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs=interp(outputs)
            # scores, predicted = outputs.max(1)
            # save top1 confidence score 
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            # top1_scores.append(probs[0][predicted.item()])
            # top1_scores.append(probs[0][predicted.item()].cpu().item())
            # max_probs,indices=torch.max(probs,dim=1)
            # # print(outputs.size())
            # print(max_probs)
            # print(indices)
            measures=torch.logical_and(probs>=0.35,probs<=0.75)
    
            uncertainty_score=torch.sum(measures,dim=(1,2,3))
            # print(uncertainty_score)
            uncertainty_scores.append(uncertainty_score.cpu().item())
            # progress_bar(idx, len(ploader))
    idx = np.argsort(uncertainty_scores)
    samples = np.array(samples)
    return samples[idx[:100]]
# entropy sampling
def get_plabels3(net, samples, cycle):
    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            e = -1.0 * torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
            top1_scores.append(e.view(e.size(0)))
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[-1000:]]

def get_classdist(samples):
    class_dist = np.zeros(10)
    for sample in samples:
        label = int(sample.split('/')[-2])
        class_dist[label] += 1
    return class_dist

if __name__ == '__main__':
    labeled = []
        
    CYCLES = 10
    for cycle in range(CYCLES):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        print('Cycle ', cycle)

        # open 5k batch (sorted low->high)
        with open(f'./loss_seg/batch_{cycle}.txt', 'r') as f:
            samples = f.readlines()
        samples=[sample[:-1] for sample in samples] # remove newline character
        if cycle > 0:
            print('>> Getting previous checkpoint')
            # prevnet = ResNet18().to(device)
            # prevnet = torch.nn.DataParallel(prevnet)
            checkpoint = torch.load(f'./checkpoint_seg/main_{cycle-1}.pth')
            net.load_state_dict(checkpoint['net'])

            # sampling
            sample100 = get_seg_plabels_confidence(net, samples, cycle)
        else:
            # first iteration: sample 1k at even intervals
            samples = np.array(samples)
            sample100 = samples[[int(j*2.7) for j in range(100)]]
        # add 1k samples to labeled set
        labeled.extend(sample100)
        print(f'>> Labeled length: {len(labeled)}')
        trainset = ActiveCityscapes(cityspace_dir,transform=image_transform, target_transform=target_transform,image_files=labeled)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
        scaler = torch.cuda.amp.GradScaler()
        wandb.log({f"{cycle}/loss"})
        for epoch in range(200):
            train(net, criterion, optimizer, epoch, trainloader)
            test(net, criterion, epoch, cycle)
            scheduler.step()
        with open(f'./main_segment_best.txt', 'a') as f:
            f.write(str(cycle) + ' ' + str(best_acc)+'\n')