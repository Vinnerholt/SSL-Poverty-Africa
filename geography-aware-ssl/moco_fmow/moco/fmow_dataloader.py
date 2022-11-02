import pandas as pd
import numpy as np
import warnings
import random
import rasterio as rio
import torch
from glob import glob

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


class CustomDatasetFromImagesTemporal(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: ALBUMENTATIONS transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = self.data_info['path'].to_numpy()
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):

        
        # Read image from tiff-file to Torch Tensor
        img_path = self.image_arr[index]
        img_src = rio.open(img_path)
        img_numpy = img_src.read()
        #img_tensor = torch.from_numpy(img_numpy)
        
        ## Select two different temporal composites
        # Select two timestamps
        t1, t2 = np.random.choice(a=10, size=2, replace=False)
        # Extract 8 corresponding bands for image 1 and 2
        img_t1 = img_numpy[8*t1 : 8*t1+8]
        img_t2 = img_numpy[8*t2 : 8*t2+8]
        
        # Transpose data from (CHW) format of PyTorch to fit (HWC) format of ALbumentations
        img_t1 = np.transpose(img_t1, (1,2,0))
        img_t2 = np.transpose(img_t2, (1,2,0))
        #print(np.shape(img_t1))
        
        # Apply transformation to images separatelyy
        img_t1 = self.transforms(image=img_t1)['image']
        img_t2 = self.transforms(image=img_t2)['image']

        # Get label(class) of the image based on the cropped pandas column
        # NOT USED WHEN NOT USING GEOLOCATION PRETEXT TASK
        single_image_label = 0

        return ([img_t1, img_t2], single_image_label)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesSpatioTemporal(Dataset):
    def __init__(self, csv_path, transform, temporal=True, spatial=True, spatial_radius=10):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: ALBUMENTATIONS transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = self.data_info['path'].to_numpy()
        # Calculate len
        self.data_len = len(self.data_info.index)

        self.spatial = spatial
        self.spatial_radius = spatial_radius
        self.temporal = temporal

        if self.spatial:
            if self.spatial_radius in range(5,16):
                column_name = 'neighbors_' + str(spatial_radius)
                neighbors_column = self.data_info[column_name]

                # Create list of lists of int indices from "stringified" lists in csv
                neighbor_list = []
                for row in neighbors_column:
                    split = row.split( )
                    ind_list = [int(ind) for ind in split[1:-1]]
                    
                    last_ind = split[-1]
                    if last_ind == '[]':
                        neighbor_list.append([])
                    else:
                        #print("last_ind: ", last_ind)
                        last_ind = last_ind.replace('[', '')
                        last_ind = last_ind.replace(']', '')
                        #print("last_ind after replace: ", last_ind, "index: ", i)
                        ind_list.append(int(last_ind))


                        neighbor_list.append(ind_list)
                
                self.neighbors = neighbor_list
            else:
                print('Spatial radius needs to be an integer in range 5-15')


    def __getitem__(self, index):
        
        #print(self.neighbors[index], len(self.neighbors[index]))
        #if self.neighbors[index] == '[]':
        #    print('empty list: ', index)
        
        # If spatial, and if image has no neighbors, reroll to select another image
        if self.spatial:
            while len(self.neighbors[index]) == 0:
                #print('rerolled index: ', index)
                index = np.random.randint(len(self.image_arr))
                #print('new index: ', index)
                


        # Read image 1
        img_path = self.image_arr[index]
        img_src = rio.open(img_path)
        img_numpy = img_src.read()
        
        # Select neighboring image for image 2
        if self.spatial:
            #print(index, " neighbors: ", np.shape(self.neighbors[index]))
            img_2_index = np.random.choice(self.neighbors[index])

            img_2_path = self.image_arr[img_2_index]
            img_2_src = rio.open(img_2_path)
            img_2_numpy = img_2_src.read()
        
        else:
            img_2_numpy = img_numpy




        ## Select two different temporal composites
        # Select two timestamps
        
        t1, t2 = np.random.choice(a=10, size=2, replace=False)
        if not self.temporal:
            t2 = t1
        # Extract 8 corresponding bands for image 1 and 2
        img_1 = img_numpy[8*t1 : 8*t1+8]
        img_2 = img_2_numpy[8*t2 : 8*t2+8]

        # Transpose data from (CHW) format of PyTorch to fit (HWC) format of ALbumentations
        img_1 = np.transpose(img_1, (1,2,0))
        img_2 = np.transpose(img_2, (1,2,0))

        
        # Apply transformation to images separatelyy
        img_1 = self.transforms(image=img_1)['image']
        img_2 = self.transforms(image=img_2)['image']

        # NOT USED WHEN NOT USING GEOLOCATION PRETEXT TASK
        single_image_label = 0

        return ([img_1, img_2], single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: ALBUMENTATIONS transforms for transforms and tensor conversion
        """


        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = self.data_info['path'].to_numpy()
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):

        img_path = self.image_arr[index]
        img_src = rio.open(img_path)
        img_numpy = img_src.read()

        t = np.random.choice(a=10, size=1)[0]
        # Extract 8 corresponding bands for image
        img_1 = img_numpy[8*t : 8*t+8]
        img_2 = np.copy(img_1)

        img_1 = np.transpose(img_1, (1,2,0))
        img_1 = self.transforms(image=img_1)['image']

        img_2 = np.transpose(img_2, (1,2,0))
        img_2 = self.transforms(image=img_2)['image']

        # NOT USED WHEN NOT USING GEOLOCATION PRETEXT TASK
        single_image_label = 0

        return ([img_1, img_2], single_image_label)

    def __len__(self):
        return self.data_len

    
