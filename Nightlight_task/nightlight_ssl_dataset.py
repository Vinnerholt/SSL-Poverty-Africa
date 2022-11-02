import pandas as pd
import numpy as np
#import warnings
import random
import rasterio as rio
import torch
#from glob import glob
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from PIL import Image

#Image.MAX_IMAGE_PIXELS = None
#warnings.simplefilter('ignore', Image.DecompressionBombWarning)


class NightlightSSLDataset(Dataset):
    def __init__(self, csv_path, csv_indices=[], transform=transforms.Compose([transforms.ToTensor()])):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        if len(csv_indices) != -0:
            self.data_info = self.data_info.iloc[csv_indices]
        
        # First column contains the image paths
        self.image_arr = [path[0] for path in self.data_info.to_numpy()]
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        
        # Get img path for MS image from index
        ms_img_path = self.image_arr[index]
        
        # Determine NL swap and get path
        nl_img_index = index
        
        # swap indicates if NL img has been swapped or not
        swap = random.randint(0,1)
        if swap:
            while nl_img_index == index:
                nl_img_index = np.random.randint(0, self.data_len)
            
        nl_img_path = self.image_arr[nl_img_index]
        
        ## Select a random temporal composite
        t = int(np.random.choice(a=10, size=1, replace=False))
        
        ms_img_src = rio.open(ms_img_path)
        ms_img_numpy = ms_img_src.read()
        
        nl_img_src = rio.open(nl_img_path)
        nl_img_numpy = nl_img_src.read()
        
        # Extract 7 MS bands for MS img and 1 NL band for NL image
        ms_img = ms_img_numpy[8*t : 8*t+7]
        #print(np.shape(ms_img))
        nl_img = nl_img_numpy[8*t+7 : 8*t+8]
        #print(np.shape(nl_img))
  
        # Concatenate MS and NL into same img and apply transformations
        img = torch.cat((torch.from_numpy(ms_img), torch.from_numpy(nl_img)), 0)
        #img = self.transforms(img)
        
        #swap = swap.to(torch.float)
        
        return (img, swap)

    def __len__(self):
        return self.data_len
