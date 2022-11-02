# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:43:29 2022

@author: Benjamin
"""

import argparse
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models


#from tensorboardX import SummaryWriter

#import moco.loader
#import moco.builder
#import datasets

IMAGE_SIZE = 224  # Input images are 224 px x 224 px
MS_CHANNELS = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NL']

class MS_NL_ResNet18(nn.Module):
    

    # SHOULD PROBABLY NOT HAVE RELU AS OUTPUT SINCE IWI CAN BE NEGATIVE FOR SOME REASON
    def __init__(self, pretrained=True, checkpoint_path=-1, num_classes=128, last_layer_activation=-1):
        super().__init__()
        
        self.ms_resnet18 = self.get_multispectral_model(pretrained, num_classes=num_classes, in_channels=len(MS_CHANNELS) - 1)
        self.nl_resnet18 = self.get_multispectral_model(pretrained, num_classes=num_classes, in_channels=1)
        
        # Check what to do with this, main_moco_tp.py applies softmax on the output
        # Do we need to specify an activation function here?
        if last_layer_activation == -1:
            print("Note: No output activation function specified.")
        self.last_layer_activation = last_layer_activation
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(512*2, num_classes)
        
        #if checkpoint_path != -1:
        #    self.ms_resnet18 = self.get_model_from_checkpoint(checkpoint_path, num_classes=num_classes)#, input_type='ms')
        #    self.nl_resnet18 = self.get_model_from_checkpoint(checkpoint_path, num_classes=num_classes)#, input_type='ms')
        
        #else:
        #    self.resnet18 = self.get_multispectral_model(pretrained, num_classes=num_classes)
        
        # else:
        #     base_resnet = models.resnet18(pretrained=False, num_classes=num_classes)
        #     base_resnet.conv1 = nn.Conv2d(len(MS_CHANNELS), 64, kernel_size=7, stride=2, padding=3,bias=False)
        
        
    # Takes MS and NL data combined as 8 channels
    def forward(self, x):
        x_ms = x[:, 0:-1, :, :]#.clone()
        x_nl = x[:, -1, :, :]#.clone()
        # Create 'channels' dimension for nl which only has 1 dimension
        x_nl = torch.unsqueeze(x_nl, 1)
        
        #print("x: ", x.size())
        #print("NL: ", x_nl.size())
        #print("MS: ", x_ms.size())
        
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        x_ms = self.ms_resnet18(x_ms)
        x_nl = self.nl_resnet18(x_nl)
        
        # Return x_ms and x_nl concatenated on top of each other
        x_cat = torch.cat((x_ms, x_nl), dim=1)
        x_cat = self.flatten(x_cat)
        x_cat = self.fc(x_cat)
        
        # Use Sigmoid for true/false model
        if self.last_layer_activation != -1:
            x_cat = self.last_layer_activation(x_cat)
        #del x, x_ms, x_nl
        
        return x_cat
    
    # Returns only a single ResNet, i.e. either MS or NL, not both
    def get_multispectral_model(self, pretrained, num_classes=128, in_channels=len(MS_CHANNELS)):
        new_in_channels = in_channels
        model = models.resnet18(pretrained=pretrained)
        
        
        
            
        
        layer = model.conv1
                
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=new_in_channels, 
                          out_channels=layer.out_channels, 
                          kernel_size=layer.kernel_size, 
                          stride=layer.stride, 
                          padding=layer.padding,
                          bias=layer.bias)
        
        
        # Copy pretrained weights from any of the RGB channels to the new MS channels
        ## vvv FROM YEH ET AL vvv
        # weights for the non-RGB channels in the first convolutional layer
        # are initialized to the mean of the weights from the RGB channels. Then all of these
        # weights are scaled by 3/C where C is the number of channels.
        # Change to this approach instead? ^
        if pretrained:
            # Initialize the weights from new channel with the red channel weights
            # Red is an arbitrary choice
            copy_weights_from_channel = 0 
            
            old_layer_weights = layer.weight.clone()
            
            # If only one channel (when creating NL model) get pretrained weights from another channel
            if in_channels == 1:
                with torch.no_grad():
                    new_layer.weight[:, 0, :, :] = old_layer_weights[:, copy_weights_from_channel, :, :]

            # Copying the weights from the old to the new layer
            # Rearranging from RGB to BGR
            elif in_channels > 3:
                with torch.no_grad():
                    

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
        
        # Remove final fc layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        
        return model
    
    # Needs to load checkpoint for both MS and NL ResNets separately
    # EASIER TO LOAD CHECKPOINT FOR ENTIRE MODEL OUTSIDE THE MODEL CLASS INSTEAD OF 
    # LOADING FOR MS AND NL RESNETS SEPARATELY IN "ms_nl_resnet.py"
    # Don't use this method
    def get_model_from_checkpoint(self, checkpoint_path, num_classes, moco=False):
        #CHECKPOINT_PATH = 'Testing/checkpoint_0009.pth.tar'

        ##### Get state for weights and optimizer from checkpoint model
        
        # Load checkpoint from previously trained model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get size of last layer in checkpoint model in order to load weights smoothly
        #checkpoint_output_size = next(reversed(checkpoint['state_dict'].items()))[1].size()[0]
        checkpoint_output_size = num_classes
        
        # Extract weights of key encoder from MoCo pretraining
        if moco:
            checkpoint_path = dict()
            for key in checkpoint['state_dict'].keys():
                if "module.encoder_k." in key:
                    resnet_key=key.replace("module.encoder_k.resnet18.", "")
                    checkpoint_path[resnet_key]=checkpoint['state_dict'][key]
        
        # E
        else:
            checkpoint_path = dict()
            for key in checkpoint['state_dict'].keys():
                resnet_key=key.replace("module.", "")
                checkpoint_path[resnet_key]=checkpoint['state_dict'][key]

        
        
        
        ##### Load new model with state from checkpoint model
        
        # Instantiate model and optimizer to dimensions of checkpoint model
        # just gets single model, not for MS and NL separately, doesn't work atm
        model = self.get_multispectral_model(pretrained=False, num_classes=checkpoint_output_size)
        

        # Load weights and optimizer from checkpoint
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        print()
        # Check keys in model
        
        
        load_result = model.load_state_dict(checkpoint_path)
        print("Model weights loading: ", load_result)
        
        
        # Adjust last layer to desired size
        # MoCo default output size is 128, but for regression we want size 1
        if num_classes != checkpoint_output_size:
            model.fc = nn.Linear(512, num_classes)
        
        return model
            
           

