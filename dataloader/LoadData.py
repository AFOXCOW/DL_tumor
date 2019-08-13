from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from skimage import img_as_float
from skimage import io
import torch
class TumorDatasetTrain(Dataset):
    
    def __init__(self, csv_file , root_dir, transform=None):
        self.labels_frame = np.array(pd.read_csv(csv_file, skiprows=1, sep=',', header=None)).astype(np.int)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, idx):
        img_name = str(idx)+'.png'
        img_path = os.path.join(self.root_dir, img_name)
        """
            !!!Pay attention!!!
            The image size is set here
            """
        img = np.empty(shape=(1, 102, 102))
        img[0, :, :] = (img_as_float(io.imread(img_path)) - 0.5)/0.5
        label = np.array([self.labels_frame[idx,1]-1])
        train_sample = {'image': img, 'label': label}
        
        if self.transform:
            train_sample = self.transform(train_sample)
        return train_sample


class TumorDatasetTest(Dataset):
    
    def __init__(self, csv_file , root_dir, transform=None):
        self.labels_frame = np.array(pd.read_csv(csv_file, skiprows= 1, sep=',', header= None)).astype(np.int)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, idx):
        img_name = str(idx) + '.png'
        img_path = os.path.join(self.root_dir, img_name)
        """
            !!!Pay attention!!!
            The image size is set here
            """
        img = np.empty(shape=(1, 102, 102))
        img[0, :, :] = (img_as_float(io.imread(img_path)) - 0.5)/0.5
        label = np.array([self.labels_frame[idx, 1]-1])
        test_sample = {'image': img, 'label': label}
        
        if self.transform:
            test_sample = self.transform(test_sample)
        return test_sample


class ToTensor(object):
    
    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        return {'image': torch.from_numpy(image), 'label': torch.LongTensor(labels)}