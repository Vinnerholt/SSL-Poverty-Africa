import pandas as pd
import numpy as np
#import warnings
#import random
#import rasterio as rio
import torch
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

from timeit import default_timer as timer

# For debugging GPU Usage
#import nvidia_smi



from torch.utils.data.dataset import Dataset
#from PIL import Image

class SupervisedDataset(Dataset):
    def __init__(self, csv_path, csv_indices=[], print_times=False):#, transform):
        """
        Args:
            csv_path (string): path to csv file
            
        """
        # Transforms
        #self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        
        if len(csv_indices) != -0:
            self.data_info = self.data_info.iloc[csv_indices]
        
        # 10th column contains image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 9])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 8], dtype=np.float32)
        # Calculate len
        self.data_len = len(self.data_info.index)
        
            # Config image parse from tfrecord
        self.bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
        self.n_temporal_frames = 10
        self.keys_to_features = {}
        for band in self.bands:
            self.keys_to_features[band] = tf.io.FixedLenFeature(shape=[224 ** 2 * self.n_temporal_frames], dtype=tf.float32)
        scalar_float_keys = ['lat', 'lon', 'year', 'iwi']
        for key in scalar_float_keys:
            self.keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
        
        # Get means and stds for all data instead of per split?
        self.band_means = [
            0.06614169065743208,
            0.09635988259340068,
            0.11381173286380034,
            0.2403052018705142,
            0.169467635861788,
            281.2976763110107,
            0.24095368620456603,
            13.069690484896771,
            5.075756473790338]

        self.band_stds = [
            0.03512808462999218,
            0.050729950158437294,
            0.07239252056994558,
            0.12444550248291769,
            0.10850729467914068,
            71.28094889057179,
            0.09077465132496389,
            18.566259343422725,
            15.861609457767129
            ]
        
        self.print_times = print_times
        
        

        #nvidia_smi.nvmlInit()
        #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        #print("SupervisedDataset __init__()")
        #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("Total memory:", info.total)
        #print("Free memory:", info.free)
        #print("Used memory:", info.used)
        #print()
        

    def __getitem__(self, index):

        # Read TFRecord file from path in csv
        img_path = self.image_arr[index]
        
        if self.print_times:
            timer1 = timer()

        
        dataset = tf.data.TFRecordDataset(
            filenames=img_path,
            compression_type='GZIP')
        
        #nvidia_smi.nvmlInit()
        #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        
        #print("create tfrecord dataset")
        #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("Total memory:", info.total)
        #print("Free memory:", info.free)
        #print("Used memory:", info.used)
        #print()

        
        if self.print_times:
            timer2 = timer()
            print("Create dataset: ", timer2 - timer1)

        # parse_example or parse_single_example? Seems to be no difference in time
        for i, example_proto in enumerate(dataset):
           #print("enumerate tfrecord dataset")
           #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
           #print("Total memory:", info.total)
           #print("Free memory:", info.free)
           #print("Used memory:", info.used)
           #print()
            
            ex = tf.io.parse_example(example_proto, features=self.keys_to_features)
        
        #print("parse example")
        #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("Total memory:", info.total)
        #print("Free memory:", info.free)
        #print("Used memory:", info.used)
        #print()
        
        
        if self.print_times:
            timer1 = timer()
            print("Parse single example: ", timer1 - timer2)
        
        # Reshape to (10, 224, 224)
        for band in self.bands:
            ##print(np.shape(ex[band]))
            ##info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            ##print("Used memory:", band, ": ", info.used)
            
            ############# TAKES UP 15 GB of GPU memory if not run on cpu
            with tf.device('/cpu:0'):
                ex[band] = tf.reshape(ex[band], (self.n_temporal_frames, 224, 224))
        
        ##print("reshape ex")
        ##info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        ##print("Total memory:", info.total)
        ##print("Free memory:", info.free)
        ##print("Used memory:", info.used)
        ##print()
        
        if self.print_times:
            timer2 = timer()
            print("Reshape bands: ", timer2 - timer1)
        
        # Convert image data of TFRecord to numpy
        # ALL COMPOSITES (80,224,224)
#         img_data = np.empty((8,10,224,224))
#         for i, k in enumerate(ex):
#             band_index = self.bands.index(k)
            
#             # Normalize data
#             if k == 'NIGHTLIGHTS': 
#                 if ex['year'] > 2015:
#                     ex[k] = (ex[k] - self.band_means[8]) / self.band_stds[8]
#                     img_data[band_index] = ex[k].numpy()
#                 else:
#                     ex[k] = (ex[k] - self.band_means[7]) / self.band_stds[7]
#                     img_data[band_index] = ex[k].numpy()

#             else:
#                 ex[k] = (ex[k] - self.band_means[band_index]) / self.band_stds[band_index]
#                 img_data[band_index] = ex[k].numpy()

#             # Only get first 8 entries, i.e. all image bands
#             if i >= 7:
#                 break
    
        # Convert image data of specific composite from TFrecord to np.array
        # ONE COMPOSITE: (8,224,224)
        img_data = np.empty((8,224,224))
        with tf.device('/cpu:0'):
            for i, k in enumerate(ex):
                #print("enumerate example proto, index: ", i, " key: ", k)
                #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                #print("Total memory:", info.total)
                #print("Free memory:", info.free)
                #print("Used memory:", info.used)
                #print()

                band_index = self.bands.index(k)
                composite_index = self.year_to_index(ex['year'])

                # Normalize data

                if k == 'NIGHTLIGHTS': 
                    if ex['year'] > 2015:
                        composite = (ex[k][composite_index] - self.band_means[8]) / self.band_stds[8]
                        img_data[band_index] = composite.numpy()
                    else:
                        composite = (ex[k][composite_index] - self.band_means[7]) / self.band_stds[7]
                        img_data[band_index] = composite.numpy()

                else:
                    composite = (ex[k][composite_index] - self.band_means[band_index]) / self.band_stds[band_index]
                    img_data[band_index] = composite.numpy()

                # Only get first 8 entries, i.e. all image bands
                if i >= 7:
                    break
        
        #print("normalize")
        #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("Total memory:", info.total)
        #print("Free memory:", info.free)
        #print("Used memory:", info.used)
        #print()
        
        if self.print_times:
            timer1 = timer()
            print("Normalize and convert to numpy: ", timer1 - timer2)
                

                
        # Reshape tensor to (80,224,224)
        #img_data = img_data.reshape((80,224,224))
        
        # Convert numpy image to torch.tensor
        img_tensor = torch.from_numpy(img_data)
        img_tensor = img_tensor.float()
        #print(type(img_tensor))
        #print(img_tensor)
        
        # Get label (iwi) from csv file
        label = self.label_arr[index]
        #print(type(label))
        
        #print("create tensors")
        #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("Total memory:", info.total)
        #print("Free memory:", info.free)
        #print("Used memory:", info.used)
        #print()
        
        if self.print_times:
            timer2 = timer()
            print("Reshape and convert to tensor: ", timer2 - timer1)
            print()
        
        # Return (image, label)
        #print("Batch retreived")
        return (img_tensor, label)
        
        # Transform?? Here or in training loop
        
        


    def __len__(self):
        return self.data_len
    
    
    def year_to_index(self, year):
        year = year - 1990
        year = int(year/3)
        
        return year
