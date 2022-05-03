"""
This is a script that contains the data loaders to be used in the deep learning models.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from PIL import Image
import json
import torchvision.transforms as transforms
"""
IMAGING FLOW CYTOMETRY IMAGES DATALOADER
"""

# Single dataset
class CellDataset_single(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.root_dir = root_dir # set the root directory to the images
        self.annotations = pd.read_csv(csv_file) # open the csv file used to the image annotations
        self.transform = transform # set the image transforms
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_list = [] # create a list to hold the images in.
        for path in self.root_dir: # for path in image directory open the images and then add to image list
              img_path = os.path.join(path,self.annotations.iloc[index,0])
              image = rasterio.open(img_path).read().squeeze(0)
              if self.transform is not None:
                  img = self.transform(image)
              img_list.append(img)
        img = torch.cat(img_list,dim=0) # Concatenet the image channels
        y_label = torch.tensor(np.float(self.annotations.iloc[index,1])) # Image labels

        return (img,y_label) # return images and labels




# combined datasets

class CellDataset_combined(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.root_dir = root_dir # set the root directory to the images
        self.annotations = pd.read_csv(csv_file) # open the csv file used to the image annotations
        self.transform = transform # set the image transforms
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_list = [] # create a list to hold the images in.
        for path in self.root_dir: # for path in image directory open the images and then add to image list
              img_path = os.path.join(path,self.annotations.iloc[index,6])
              image = rasterio.open(img_path).read().squeeze(0)
              if self.transform is not None:
                  img = self.transform(image)
              img_list.append(img)
        img = torch.cat(img_list,dim=0) # Concatenet the image channels
        y_label = torch.tensor(np.float(self.annotations.iloc[index,1])) # Image labels

        return (img,y_label) # return images and labels




"""
CYTOIMAGENET DATALOADER
"""



class Cytoimagenet(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir  # Directory to the images
        self.annotations = csv_file
        self.transform = transform
        with open('label_dict.json', 'r') as fp:
            self.clas_to_dict = json.load(fp)
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_full_path = self.root_dir + self.annotations.iloc[idx, 9] + '/' + self.annotations.iloc[idx, 10]
        image = Image.open(img_full_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(image)
        label = self.annotations.iloc[idx, 21]
        # Grab the index of the label
        y_label = torch.tensor(int(self.clas_to_dict[label]))
        # Convert index to tensor.
        # Grab the index of the y label from the dict and then turn to tensor
        return (img, y_label)



