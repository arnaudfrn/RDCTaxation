import os
import shutil

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn # Add on classifier
from torch import optim # Loss and optimizer


class HouseDataset(Dataset):
    """
    Custom dataset class for Houses dataset
    Index through each tiled house and its price
    """

    def __init__(self, df_x, df_y, path, transform=None):
        self.path = path
        self.transform = transform
        self.df_x = df_x
        self.df_y = df_y

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, idx):
        # given that the response variable is the house price
        # we will pass that along with the respective house
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.path,
                                self.df_x[['image']].iloc[idx][0])
        image = io.imread(img_name)
        target = self.df_y[['price']].iloc[idx][0]
        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample
