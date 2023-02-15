'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np
import copy
from models import *
from loader import Loader, Loader2
from utils.utils import progress_bar
import model
import wandb
WANDB_KEY = "416ff8e8f97b3ca056e121705709bec3d83e929b"
WANDB_HOST = "https://api.wandb.ai"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["WANDB_API_KEY"] = WANDB_KEY
os.environ["WANDB_BASE_URL"] = WANDB_HOST
wandb.init(project="VAAL Active Learning", name="PTAL using VAE")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_vae_steps=2
num_adv_steps=1
batch_size=128
latent_dim=32
adversary_param=1
beta=1
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

testset = Loader(is_train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
def vae_loss(x, recon, mu, logvar, beta):
    MSE = nn.MSELoss()(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

# Training
def train_with_vae(task_model,vae,discriminator,criterion,optimizer,epoch, labeled_dataloader,unlabeled_dataloader):
    optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
    # optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
    task_model.train()
    vae.train()
    discriminator.train()

    print('\nEpoch: %d' % epoch)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (labeled_images, targets) in enumerate(labeled_dataloader):
        labeled_images, targets = labeled_images.to(device), targets.to(device) # labeld data
        unlabeled_images,_=next(iter(unlabeled_dataloader)) # unlabeld data
        unlabeled_images=unlabeled_images.to(device)
        optimizer.zero_grad()

        # task_model step
        outputs = task_model(labeled_images)
        task_loss = criterion(outputs, targets)
        task_loss.backward()
        optimizer.step()

        # VAE step
        for count in range(num_vae_steps):
            recon, z, mu, logvar = vae(labeled_images)
            unsup_loss = vae_loss(labeled_images, recon, mu, logvar,beta)
            unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_images)
            transductive_loss = vae_loss(unlabeled_images, 
                    unlab_recon, unlab_mu, unlab_logvar, beta=1)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_images.size(0))
            unlab_real_preds = torch.ones(unlabeled_images.size(0))
                
            # if self.args.cuda:
            lab_real_preds = lab_real_preds.to(device)
            unlab_real_preds = unlab_real_preds.to(device)

            dsc_loss = nn.BCELoss()(labeled_preds, lab_real_preds) + \
                    nn.BCELoss()(unlabeled_preds, unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            optim_vae.zero_grad()
            total_vae_loss.backward()
            optim_vae.step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_images, _ = next(iter(labeled_dataloader))
                unlabeled_images,_= next(iter(unlabeled_dataloader))
                # if self.args.cuda:
                labeled_images = labeled_images.to(device)
                unlabeled_images = unlabeled_images.to(device)

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_images)
                _, _, unlab_mu, _ = vae(unlabeled_images)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_images.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_images.size(0))

            # if self.args.cuda:
            lab_real_preds = lab_real_preds.to(device)
            unlab_fake_preds = unlab_fake_preds.to(device)
            
            dsc_loss = nn.BCELoss()(labeled_preds, lab_real_preds) + \
                   nn.BCELoss()(unlabeled_preds, unlab_fake_preds)

            optim_discriminator.zero_grad()
            dsc_loss.backward()
            optim_discriminator.step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps - 1):
                labeled_images, _ = next(iter(labeled_dataloader))
                unlabeled_images,_ = next(iter(unlabeled_dataloader))

                # if self.args.cuda:
                labeled_images = labeled_images.to(device)
                unlabeled_images = unlabeled_images.to(device)

            

        if batch_idx % 100 == 0:
            print('Current training iteration: {}'.format(batch_idx))
            print('Current task model loss: {:.4f}'.format(task_loss.item()))
            print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
            print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

        # if batch_idx % 1000 == 0:
        #     acc = self.validate(task_model, val_dataloader)
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_model = copy.deepcopy(task_model)
            
        #     print('current step: {} acc: {}'.format(batch_idx, acc))
        #     print('best acc: ', best_acc)

        train_loss += task_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(labeled_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return task_model,vae, discriminator
def test(net, criterion, epoch, cycle):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
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
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/main_{cycle}.pth')
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
## return 1k samples w/ lowest top1 score
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
            top1_scores.append(probs[0][predicted.item()])
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:1000]]

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

def get_plabels_by_vae(vae, discriminator,samples,cycle):
    # sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    unlabel_scores = []
    vae.eval()
    discriminator.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, mu, _ = vae(inputs)
            preds = discriminator(mu)
            # preds = preds.cpu().data
            unlabel_scores.append(preds.view(preds.size(0)).item())
            progress_bar(idx, len(ploader))
    idx=np.argsort(unlabel_scores)
    samples=np.array(samples)
    return samples[idx[-1000:]]

def get_classdist(samples):
    class_dist = np.zeros(10)
    for sample in samples:
        label = int(sample.split('/')[-2])
        class_dist[label] += 1
    return class_dist

if __name__ == '__main__':
    labeled = []
    vae = model.VAE(latent_dim)
    discriminator = model.Discriminator(latent_dim)
    vae=vae.to(device)
    discriminator=discriminator.to(device)      
    CYCLES = 10
    for cycle in range(CYCLES):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        print('Cycle ', cycle)

        # open 5k batch (sorted low->high)
        with open(f'./loss/batch_{cycle}.txt', 'r') as f:
            samples = f.readlines()
        # create new discriminator and VAE
 
        if cycle > 0:
            print('>> Getting previous checkpoint')
            # prevnet = ResNet18().to(device)
            # prevnet = torch.nn.DataParallel(prevnet)
            checkpoint = torch.load(f'./checkpoint/main_{cycle-1}.pth')
            net.load_state_dict(checkpoint['net'])

            # sampling
            sample1k = get_plabels_by_vae(vae, discriminator,samples,cycle)
        else:
            # first iteration: sample 1k at even intervals
            samples = np.array(samples)
            sample1k = samples[[j*5 for j in range(1000)]]
        # add 1k samples to labeled set
        labeled.extend(sample1k)
        unlabeled_samples=[sample for sample in samples if sample not in labeled]
        print(f'>> Labeled length: {len(labeled)}')
        labeled_trainset = Loader2(is_train=True, transform=transform_train, path_list=labeled)
        labelded_trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        unlabeled_trainset = Loader2(is_train=True, transform=transform_train, path_list=unlabeled_samples)
        unlabelded_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        for epoch in range(200):
            task_model,vae,discriminator=train_with_vae(net, vae,discriminator, criterion, optimizer, epoch, labelded_trainloader,unlabelded_trainloader)
            test(task_model, criterion, epoch, cycle)
            scheduler.step()
        wandb.log({'Accuracy':best_acc})
        with open(f'./main_best.txt', 'a') as f:
            f.write(str(cycle) + ' ' + str(best_acc)+'\n')
