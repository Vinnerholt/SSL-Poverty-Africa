
import torch
import pandas as pd
import numpy as np
import pickle as pkl
import os
import glob

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#%matplotlib inline

import supervised_dataset
import importlib
importlib.reload(supervised_dataset)

import sys
sys.path.insert(0, '/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/geography-aware-ssl/moco_fmow/moco')
from ms_nl_resnet import MS_NL_ResNet18
  
# Load model from checkpoint and evaluate on test set from FOLD
def evaluate_model(checkpoints_dir, fold, fraction, gpu=0, csv="/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised/dhs_clusters_paths.csv", cap_at_100=True, plot_title='', file_name=''):
    
    # Get folds path from fraction
    folds_paths = dict()
    folds_paths['1'] = '/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/CreateFolds/new_dhs_incountry_folds.pkl'
    fracs = ['0_01', '0_05', '0_1']
    for frac in fracs:
        folds_paths[frac] = '/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/CreateFolds/dhs_incountry_folds_{}.pkl'.format(frac)
    folds_path = folds_paths[fraction]
 

    ### Read loss files for fold and find best epoch
    print("FOLD: ", fold)
    fold_path = os.path.join(checkpoints_dir, 'fold_{}'.format(fold))

    fold
    
    loss_files = []
    for file in os.listdir(fold_path):
        if 'losses' in file:
            loss_files.append(file)
    loss_files.sort()
    last_loss_file = loss_files[-1]
    
    losses = pd.read_csv(os.path.join(fold_path, last_loss_file), sep=",", header=None)
    best_epoch = losses[1].argmin()
    print("Best epoch: ", best_epoch)
    print("Best epoch val_loss: ", losses[1].iloc[best_epoch])
    ###                    
    
    ### Create model and load checkpoint from best epoch
    model = MS_NL_ResNet18(num_classes=1, last_layer_activation = torch.nn.ReLU())
                        
    checkpoint_path = os.path.join(fold_path, 'checkpoint_fold_{}_{:04d}.pth.tar'.format(fold, best_epoch))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                            
    checkpoint_dict = dict()
    for key in checkpoint['state_dict'].keys():
        resnet_key=key.replace("module.", "")
        checkpoint_dict[resnet_key]=checkpoint['state_dict'][key]
            
    model.load_state_dict(checkpoint_dict)
    print("Model loaded: ", checkpoint_path)
    
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    ###

    ### Read folds and create dataloader
    with open(folds_path, 'rb') as handle:
        folds = pkl.load(handle)
    
    test_dataset = supervised_dataset.SupervisedDataset(csv, csv_indices=folds[fold]['test'])

    batch_size = 64
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    ###


    ### Evaluate model on test set
    iwi_pred = []
    iwi_true = []

    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda(gpu)
        labels = labels.cuda(gpu)
        
        preds = model(images)
        preds = torch.squeeze(preds)
        
        for pred in preds:
            iwi_pred.append(pred.item())
        
        for label in labels:
            iwi_true.append(label.item())
        
        # Print every 20th batch
        if i % 20 == 0:
            print(i, "/", len(test_loader))
    ###

    # Cap IWI predictions at 100 if specified
    if cap_at_100:
        for j, iwi in enumerate(iwi_pred):
            if iwi > 100:
                iwi_pred[j] = 100


    ### Calculate r^2 and plot scatter plot with regression line
    r2 = r2_score(iwi_true, iwi_pred)
    print("r2: ", r2)

    plt.figure(figsize=(10,10))
    plt.scatter(iwi_true, iwi_pred, alpha=0.05, c='steelblue')
    m, b = np.polyfit(iwi_true, iwi_pred, 1)

    x = [i for i in range(101)]
    y = [i*m + b for i in x]

    print("slope: ", m, "intercept: ", b)

    plt.title(plot_title + '\n ' + '$R^2$ = ' + str(round(r2, 4)) + '\n' + 'fold: ' + str(fold), fontsize=15)
    plt.xlabel('True IWI', fontsize=20)
    if cap_at_100:
        plt.ylabel('Predicted IWI (capped at 100)', fontsize=20)
    else:
        plt.ylabel('Predicted IWI', fontsize=20)
    plt.plot(x, y, c='red')
    plt.show()
    
    plt.savefig('r2_plot_{}_fold_{}.png'.format(file_name, fold))
    ###
    
    with open(checkpoints_dir + '/preds_{}.pkl'.format(fold), 'wb') as f:
        pkl.dump(iwi_pred, f)


    return r2






#_______________________________________

def evaluate_model_all_folds(checkpoints_dir, folds_path, gpu=0, new_txt_format=True, csv="/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/Supervised/dhs_clusters_paths.csv", cap_at_100=True, plot_title='', file_name=''):
    #FOLDS = ['A', 'B', 'C', 'D', 'E']
    # Get folds path from fraction
    # folds_paths = dict()
    # folds_paths['1'] = '/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/CreateFolds/new_dhs_incountry_folds.pkl'
    # fracs = ['0_01', '0_05', '0_1']
    # for frac in fracs:
    #     folds_paths[frac] = '/cephyr/NOBACKUP/groups/globalpoverty1/JesperBenjamin/CreateFolds/dhs_incountry_folds_{}.pkl'.format(frac)
    # folds_path = folds_paths[fraction]
 
    

    ### Read loss files for fold and find best epoch
    #
    
    ###
    
        ### Read folds and create dataloader
    with open(folds_path, 'rb') as handle:
        folds = pkl.load(handle)

    FOLDS = folds.keys()

     ### Evaluate model on test set
    iwi_pred = []
    iwi_true = []

    
    for fold in FOLDS:
        print("FOLD: ", fold)
        fold_path = os.path.join(checkpoints_dir, 'fold_{}'.format(fold))

        loss_files = []
        for file in os.listdir(fold_path):
            if 'losses' in file:
                loss_files.append(file)
        loss_files.sort()
        
        if new_txt_format == False:
            last_loss_file = loss_files[-1]

            losses = pd.read_csv(os.path.join(fold_path, last_loss_file), sep=",", header=None)
            best_epoch = losses[1].argmin()
            print("Best epoch: ", best_epoch)
            print("Best epoch val_loss: ", losses[1].iloc[best_epoch])
        
        elif new_txt_format == True:
            last_loss_files = loss_files[-4:]
            loss_gpus = []
            for i, loss_gpu in enumerate(last_loss_files):
                loss_df = (pd.read_csv(os.path.join(fold_path, loss_gpu), sep=",", header=None))
                loss_gpus.append(loss_df[1].values)

            mean_losses = np.mean(loss_gpus, axis=0)
            best_epoch = mean_losses.argmin()
            
            print("Best epoch: ", best_epoch)
            print("Best epoch val_loss: ", mean_losses[best_epoch])
        ###                    

        ### Create model and load checkpoint from best epoch
        model = MS_NL_ResNet18(num_classes=1, last_layer_activation = torch.nn.ReLU())
        

        checkpoint_path = os.path.join(fold_path, 'checkpoint_fold_{}_{:04d}.pth.tar'.format(fold, best_epoch))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        checkpoint_dict = dict()
        for key in checkpoint['state_dict'].keys():
            resnet_key=key.replace("module.", "")
            checkpoint_dict[resnet_key]=checkpoint['state_dict'][key]

        model.load_state_dict(checkpoint_dict)
        print("Model loaded: ", checkpoint_path)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)

        test_dataset = supervised_dataset.SupervisedDataset(csv, csv_indices=folds[fold]['test'])

        batch_size = 64
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        ###


        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda(gpu)
            labels = labels.cuda(gpu)

            preds = model(images)
            preds = torch.squeeze(preds)

            for pred in preds:
                iwi_pred.append(pred.item())

            for label in labels:
                iwi_true.append(label.item())

            # Print every 20th batch
            if i % 20 == 0:
                print(i, "/", len(test_loader))
        ###

    # Cap IWI predictions at 100 if specified
    if cap_at_100:
        for j, iwi in enumerate(iwi_pred):
            if iwi > 100:
                iwi_pred[j] = 100


    ### Calculate r^2 and plot scatter plot with regression line
    r2 = r2_score(iwi_true, iwi_pred)
    print("r2: ", r2)

    plt.figure(figsize=(10,10))
    plt.scatter(iwi_true, iwi_pred, alpha=0.05, c='steelblue')
    m, b = np.polyfit(iwi_true, iwi_pred, 1)

    x = [i for i in range(101)]
    y = [i*m + b for i in x]

    print("slope: ", m, "intercept: ", b)

    plt.title(plot_title + '\n ' + '$R^2$ = ' + str(round(r2, 4)), fontsize=15)
    plt.xlabel('True IWI', fontsize=20)
    if cap_at_100:
        plt.ylabel('Predicted IWI (capped at 100)', fontsize=20)
    else:
        plt.ylabel('Predicted IWI', fontsize=20)
    plt.plot(x, y, c='red')
    plt.show()
    
    plt.savefig('r2_plot_{}.png'.format(file_name))
    ###
    
    with open(checkpoints_dir + '/preds_{}.pkl'.format(file_name), 'wb') as f:
        pkl.dump(iwi_pred, f)


    return r2