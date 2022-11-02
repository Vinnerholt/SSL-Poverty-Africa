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

#import gc

import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


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

sys.path.insert(0, '/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/geography-aware-ssl/moco_fmow/moco')
from ms_nl_resnet import MS_NL_ResNet18
from supervised_dataset import SupervisedDataset



if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    import time

    #del train_set, train_loader, model, gpu, criterion, epoch, writer, optimizer
    torch.cuda.empty_cache()

    train_set = SupervisedDataset('dhs_clusters_paths.csv')#, print_times=True)
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=32,
            drop_last=True
    )

    model = MS_NL_ResNet18(num_classes=1)

    print("CUDA: ", torch.cuda.is_available())

    gpu = 0

    criterion = torch.nn.MSELoss().cuda(gpu)

    epoch=1

    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    ############
    torch.cuda.set_device(gpu)
    model.cuda(gpu)


    print()

    
    epochs = 100
    losses = [[] for x in range(epochs)]
    
    model.train()
#n_iters = len(train_loader) * epoch

    

    end = time.time()
    for epoch in range(epochs):

        for i, (images, labels) in enumerate(train_loader):
            # measure data loading time
            print("batch: ", i)
            #data_time.update(time.time() - end)
            print()
            #print("start")
            #!nvidia-smi

            #if args.gpu is not None:
            images = images.cuda(gpu)#, non_blocking=True)

            #print()
            #print("images to cuda")
            #!nvidia-smi

            labels = labels.cuda(gpu)#, non_blocking=True)
            #print()
            #print("labels to cuda")
            #!nvidia-smi
                #images[1] = images[1].cuda(args.gpu, non_blocking=True)

            #print(labels)
            # compute output
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
            optimizer.zero_grad()

            output = model(images)
            #print("pass images through model")
            #print()
            #!nvidia-smi
           # print("Output: ", output.size(), "Labels: ", labels.size())
            output = torch.squeeze(output)
            #print("Output: ", output.size(), "Labels: ", labels.size())
            loss = criterion(output, labels)

            print("LOSS: ", loss)

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            #print(loss.size())
            losses[epoch].append(loss.detach().item())

            # measure elapsed time
            #batch_time.update(time.time() - end)
            print(time.time() - end)
            #end = time.time()

            #!nvidia-smi
            #if i % args.print_freq == 0:
            #    progress.display(i)
        #
            #    if writer is not None:
            #        writer.add_scalar('pretrain/acc1', top1.avg, n_iters+i)
            #        writer.add_scalar('pretrain/acc5', top5.avg, n_iters+i)
            #        writer.add_scalar('pretrain/batch_time',
            #                          batch_time.avg, n_iters+i)
            #        writer.add_scalar('pretrain/data_time',
            #                          data_time.avg, n_iters+i)
            #        writer.add_scalar('pretrain/loss', losses.avg, n_iters+i)
            #        writer.add_scalar('pretrain/top_n', top_n.avg, n_iters+i)
        #

        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 'checkpoint_supervised_large_40h_{:04d}.pth.tar'.format(epoch))


    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 'checkpoint_supervised_large_final_40h{:04d}.pth.tar'.format(epoch))
    
    epoch_losses = []

    textfile = open("losses_large_40h.txt", "w")
    for epoch in losses:
        if len(epoch) != 0:
            mean_loss = np.mean(epoch)
            epoch_losses.append(mean_loss)
            textfile.write(str(mean_loss) + "\n")
        #for element in epoch:
        #    textfile.write(str(element) + "\n")
        #textfile.write("\n")
    textfile.close()
    print("EPOCH LOSSES: ", epoch_losses)