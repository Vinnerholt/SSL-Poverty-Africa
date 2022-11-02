import pandas as pd
import numpy as np
import warnings
import random
import torch
from torch.utils.data.dataset import Dataset

from batchers import batcher

# IDEA: TFRecord -> Numpy Arrays -> Pytorch tensors 
# Necessity: 
# Need to keep CSV info for each tensor
Class TrainingDataset(Dataset):
    
    def __init__(self, csv_path):
        self.data_info=pd.read_csv(csv_path + '/dhs_clusters.csv')
        dhs_clusters=dhs_clusters.sort_values(['country', ' year']).reset_index(drop=True)
        
        
    def ReadTFrecord(self):
        tfrecord_files = np.asarray(batcher.create_full_tfrecords_paths(csv_path))
        dhs_clusters['file']=tfrecord_files
        
        return dhs_clusters
    def CustomDataset(self):
        ds= batcher.get_dataset(dhs_clusters['file'].values, 0, labeled=False, size_of_window=10, one_year_model=False, n_year_composites=10, normalize=False, max_epochs=1)
        for d in ds.as_numpy_iterator():
            t = torch.from_numpy(d['model_input'])
            
            return t
            
        
    