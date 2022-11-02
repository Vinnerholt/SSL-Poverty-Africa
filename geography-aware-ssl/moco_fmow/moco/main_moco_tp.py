#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
#import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tensorboardX import SummaryWriter

import loader
import builder
#import ms_resnet
#import datasets

import nvidia_smi

sys.path.insert(0, '/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/geography-aware-ssl/moco_fmow/moco')
from ms_nl_resnet import MS_NL_ResNet18

from fmow_dataloader import CustomDatasetFromImages, CustomDatasetFromImagesTemporal, CustomDatasetFromImagesSpatioTemporal

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ms_nl_resnet18',
                    #choices=model_names,
                    help='ONLY ms_resnet18 SUPPORTED FOR MULTISPECTRAL. model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: ms_nl_resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--loss', type=str, default='cpc',
                    choices=['cpc', 'ml_cpc'],
                    help="cpc: original objective, \
                          ml_cpc: multi-class, alpha weights extension")

parser.add_argument('--ml-cpc-alpha', default=1.0, type=float,
                    help='alpha hyperparameter for ml_cpc')
parser.add_argument('--ml-cpc-alpha-low', default=1.0, type=float)
parser.add_argument('--ml-cpc-alpha-geo', action='store_true',
                    help='use geometric scheduling for alpha in ml_cpc')

parser.add_argument('--save-dir', default='', type=str,
                    help='save location')
parser.add_argument('--ckpt_frequency', default=1, type=int,
                    help='frequency of updating checkpoint')

parser.add_argument('--augs', default='moco', type=str,
                    help='which set of augmentations to use')
parser.add_argument('--pretrained', action='store_true',
                    help='use model pretrained on imagenet?')
parser.add_argument('--temporal', action='store_true',
                    help='use temporal pairs')
parser.add_argument('--spatial', action='store_true',
                    help='use spatial pairs')
parser.add_argument('--spatial-radius', default=10, type=int,
                    help='radius for neighbors')

# CHANGE train_csv TO CHANGE TRAINING FILES
train_csv = "/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/imagepaths_all_neighbors.csv"
#temporal = True

def main():
    print('In main')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        if len(args.save_dir) > 0:
            os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    print('Check if MS-ResNet18 is specified')
    ## Should only run for our model
    if args.arch not in ['ms_resnet18', 'ms_nl_resnet18'] :
        raise NotImplementedError("Only MS-ResNet18 and MS-NL-ResNet18 is supported for multispectral data.")

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("args.distributed: " + str(args.distributed))
    print("MLP: " + str(args.mlp))
    print("Pretrained?: ", args.pretrained)
    print("Temporal: ", args.temporal)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        #print('Attempting mp.spawn')
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu
    
    #print('in main_worker method')
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        #builtins.print = print_pass
   
    #print(1)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    #print(2)
    if args.distributed:
        #print("distributed 1")
        #print(3)
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            #print("distributed 2")
        #print(4)
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
	    #print(args.rank)
            args.rank = args.rank * ngpus_per_node + gpu
	    #print(args.rank)
        #print('attempting dist.init_process_group')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
	#print("hej funkar detta")
    # create model
    print("=> creating model '{}'".format(args.arch))
    
    # Build MS-Resnet18
    model = builder.MoCo(
        args.arch,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.pretrained)

    #nvidia_smi.nvmlInit()
    #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    #print("Models built)")
    #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    #print("Total memory:", info.total)
    #print("Free memory:", info.free)
    #print("Used memory:", info.used)
    #print()
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
                                                   
            #nvidia_smi.nvmlInit()
            #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            #print("Models to CUDA")
            #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            #print("Total memory:", info.total)
            #print("Free memory:", info.free)
            #print("Used memory:", info.used)
            #print()
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    if args.rank == 0:
        writer = SummaryWriter(logdir=args.save_dir)
    else:
        writer = None
    # define loss function (criterion) and optimizer
    if args.loss == 'cpc':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss == 'ml_cpc':
        criterion = nn.KLDivLoss(reduction='batchmean').cuda(args.gpu)
    else:
        raise NotImplementedError(args.loss + ' not implemented yet.')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ## @@@ MULTISPECTRAL TODO
    #normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                                 std=(0.229, 0.224, 0.225))
#
    #if args.aug_plus:
    #    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    #    augmentation = [
    #        # Don't want to resize or crop?
    #        #transforms.Resize(224*2),
    #        #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    #        transforms.RandomApply([
    #            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    #        ], p=0.8),
    #        transforms.RandomGrayscale(p=0.2),
    #        transforms.RandomApply(
    #            [loader.GaussianBlur([.1, 2.])], p=0.5),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize
    #    ]
    #else:
    #    pass
        #raise NotImplementedError("Moco v1 augmentation not adapted")


    if args.augs == 'moco':
        aug = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
            A.GaussianBlur(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ])
    elif args.augs == 'flips':
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ])
    else:
        aug = A.Compose([
            ToTensorV2()
        ])

    # Not temporal
    if args.temporal == False:
        # Baseline
        #train_dataset = CustomDatasetFromImages(train_csv,
        #    transform=loader.TwoCropsTransform(
        #        transforms.Compose(augmentation)))

        train_dataset = CustomDatasetFromImages(train_csv,
            transform=aug)

    
    elif args.spatial == True:
        train_dataset = CustomDatasetFromImagesSpatioTemporal(train_csv,
            transform=aug, temporal=args.temporal, spatial=args.spatial, spatial_radius=args.spatial_radius)
    
    # Temporal
    else:  
        train_dataset = CustomDatasetFromImagesTemporal(train_csv,
            transform=aug)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if epoch % args.ckpt_frequency == 0 or epoch == args.epochs - 1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch)))              

    if writer is not None:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top_n = AverageMeter('Acc@n', ':6.2f')

    progress_items = [batch_time, data_time, losses, top1, top5]
    progress_items += [top_n]

    #print(torch.cuda.list_gpu_processes())
    
    progress = ProgressMeter(
        len(train_loader),
        progress_items,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    n_iters = len(train_loader) * epoch

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)


        output, target = model(im_q=images[0], im_k=images[1])

        if args.loss == 'cpc':
            loss = criterion(output, target)
        elif args.loss == 'ml_cpc':
            device = output.device
            k = output.size(1)
            if args.ml_cpc_alpha_geo:
                alpha = (np.log(args.ml_cpc_alpha) - np.log(args.ml_cpc_alpha_low)) * (1 - float(epoch) / args.epochs) + np.log(args.ml_cpc_alpha_low)
                alpha = np.exp(alpha)
            else:
                alpha = args.ml_cpc_alpha
            beta = (k - alpha) / (k - 1)
            logits = torch.cat([
                output[:, :1] + torch.log(torch.tensor(alpha).float()).to(device),
                output[:, 1:] + torch.log(torch.tensor(beta).float()).to(device)
            ], dim=1)
            logits = logits.view(1, -1)
            labels = torch.zeros_like(output)
            labels[:, 0] += 1.0 / output.size(0)
            labels = labels.view(1, -1)
            loss = criterion(F.log_softmax(logits, dim=1), labels)
            loss = loss / torch.tensor(alpha).float()
        else:
            raise NotImplementedError(args.loss + ' not implemented.')

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        acc_n = accuracy_n(output)
        top_n.update(acc_n.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if writer is not None:
                writer.add_scalar('pretrain/acc1', top1.avg, n_iters+i)
                writer.add_scalar('pretrain/acc5', top5.avg, n_iters+i)
                writer.add_scalar('pretrain/batch_time',
                                  batch_time.avg, n_iters+i)
                writer.add_scalar('pretrain/data_time',
                                  data_time.avg, n_iters+i)
                writer.add_scalar('pretrain/loss', losses.avg, n_iters+i)
                writer.add_scalar('pretrain/top_n', top_n.avg, n_iters+i)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if args.loss == 'mw_cpc':
        weight_decay = args.weight_decay
        if args.ml_cpc_alpha_cos:
            alpha = (args.ml_cpc_alpha - args.ml_cpc_alpha_low) * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) + args.ml_cpc_alpha_low
        elif args.ml_cpc_alpha_geo:
            alpha = (np.log(args.ml_cpc_alpha) - np.log(args.ml_cpc_alpha_low)) * (1 - float(epoch) / args.epochs) + np.log(args.ml_cpc_alpha_low)
            alpha = np.exp(alpha)
        else:
            alpha = args.ml_cpc_alpha
        print('weight decay {}'.format(weight_decay * alpha))
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay * alpha



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_n(output):
    """
    Computes the accuracy over the top n predictions for Nx(K+1) class, N label classification.
    We assume that the targets are size (N, K+1) where the first element is positive.
    """
    with torch.no_grad():
        top_n = output.size(0)
        k = output.size(1)
        _, pred = output.view(-1).topk(top_n, 0, True, True)
        correct = pred.fmod(k).eq(0.0)
        return correct.float().sum().mul(100.0 / top_n)

if __name__ == '__main__':
        
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    print("START")
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)
    print()
    
    main()

