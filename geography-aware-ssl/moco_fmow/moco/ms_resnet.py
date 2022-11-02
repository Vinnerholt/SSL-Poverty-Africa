# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:43:29 2022

@author: Benjamin
"""

import argparse
#import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

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

#from tensorboardX import SummaryWriter

#import moco.loader
#import moco.builder
#import datasets

IMAGE_SIZE = 224  # Input images are 224 px x 224 px
MS_CHANNELS = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NL']

class MSResNet18(nn.Module):
    

    
    def __init__(self, pretrained=True, checkpoint_path=-1, num_classes=128):
        super().__init__()
        
        if checkpoint_path != -1:
            self.resnet18 = self.get_model_from_checkpoint(checkpoint_path, num_classes=num_classes)
        
        else:
            self.resnet18 = self.get_multispectral_model(pretrained, num_classes=num_classes)
        
        # else:
        #     base_resnet = models.resnet18(pretrained=False, num_classes=num_classes)
        #     base_resnet.conv1 = nn.Conv2d(len(MS_CHANNELS), 64, kernel_size=7, stride=2, padding=3,bias=False)
        
        
        
    def forward(self, x):
        x = self.resnet18(x)
        return x
    
    def get_multispectral_model(self, pretrained, num_classes=128):
        new_in_channels = len(MS_CHANNELS)
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes) # FINAL FC LAYER modification
        
        layer = model.conv1
                
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=new_in_channels, 
                          out_channels=layer.out_channels, 
                          kernel_size=layer.kernel_size, 
                          stride=layer.stride, 
                          padding=layer.padding,
                          bias=layer.bias)
        
        
        # Copy pretrained weights from any of the RGB channels to the new MS channels
        if pretrained:
            # Initialize the weights from new channel with the red channel weights
            # Red is an arbitrary choice
            copy_weights_from_channel = 0 

            # Copying the weights from the old to the new layer
            # Rearranging from RGB to BGR
            with torch.no_grad():
                old_layer_weights = layer.weight.clone()

                new_layer.weight[:, 0, :, :] = old_layer_weights[:, 2, :, :] #Blue
                new_layer.weight[:, 1, :, :] = old_layer_weights[:, 1, :, :] #Green
                new_layer.weight[:, 2, :, :] = old_layer_weights[:, 0, :, :] #Red

            #Copying the weights of the `copy_weights_from_channel` channel of the old layer to the extra (MS) channels of the new layer
            for i in range(new_in_channels - layer.in_channels):
                channel = layer.in_channels + i
                with torch.no_grad():
                    new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights_from_channel:copy_weights_from_channel+1, : :].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)
        
        model.conv1 = new_layer
        
        return model
    
    def get_model_from_checkpoint(self, checkpoint_path, num_classes):
        #CHECKPOINT_PATH = 'Testing/checkpoint_0009.pth.tar'

        ##### Get state for weights and optimizer from checkpoint model
        
        # Load checkpoint from previously trained model
        checkpoint = torch.load(checkpoint_path)
        
        # Get size of last layer in checkpoint model in order to load weights smoothly
        checkpoint_output_size = next(reversed(checkpoint['state_dict'].items()))[1].size()[0]
        
        # Extract weights of key encoder from MoCo pretraining
        encoder_k_state_dict = dict()
        for key in checkpoint['state_dict'].keys():
            if "module.encoder_k." in key:
                resnet_key=key.replace("module.encoder_k.resnet18.", "")
                encoder_k_state_dict[resnet_key]=checkpoint['state_dict'][key]

        # Extract optimizer state from MoCo pretraining
        # optimizer_state_dict = checkpoint['optimizer']
        # optimizer_state_dict['param_groups'][0]['params'] = [i for i in range(62)]
        # optimizer_state_dict['param_groups'][0]['params'] 
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        # optimizer.load_state_dict(optimizer_state_dict)
        
        
        
        ##### Load new model with state from checkpoint model
        
        # Instantiate model and optimizer to dimensions of checkpoint model
        model = self.get_multispectral_model(pretrained=False, num_classes=checkpoint_output_size)
        

        # Load weights and optimizer from checkpoint
        load_result = model.load_state_dict(encoder_k_state_dict)
        print("Model weights loading: ", load_result)
        
        
        # Adjust last layer to desired size
        # MoCo default output size is 128, but for regression we want size 1
        if num_classes != checkpoint_output_size:
            model.fc = nn.Linear(512, num_classes)
        
        return model
            
            ## FIX STATE OF OPTIMIZER FOR LAST LAYER??

