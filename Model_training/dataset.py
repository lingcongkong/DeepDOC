
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset


class CustomDataset(Dataset):
    def __init__(self):
        tb = pd.read_csv('/share/home/konglingcong/vegi/doc_data/table/tb_MCSVS.csv', index_col=0)
        self.data = []
        self.length = []

        for i in tb.file:
            img = nib.load(i)
            img = np.transpose(np.asanyarray(img.dataobj), (3, 0, 1, 2))
            img = img
            img = torch.tensor(img, dtype=torch.float32)
            self.data.append(img)
            self.length.append(img.shape[0])
        self.labels = tb.type
        self.labels = torch.tensor(self.labels, dtype=torch.int8)
        print(self.length)
        # self.data = torch.concat(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class CustomDataset2(Dataset):
    def __init__(self):
        tb = pd.read_csv('/share/home/konglingcong/vegi/doc_data/table/tb_HCDOC.csv', index_col=0)
        self.data = []
        self.length = []

        for i in tb.file:
            img = nib.load(i)
            img = np.transpose(np.asanyarray(img.dataobj), (3, 0, 1, 2))
            img = img
            img = torch.tensor(img, dtype=torch.float32)
            self.data.append(img)
            self.length.append(img.shape[0])
        self.labels = tb.type
        self.labels = torch.tensor(self.labels, dtype=torch.int8)
        print(self.length)
        # self.data = torch.concat(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CustomDataset3(Dataset):
    def __init__(self):
        tb = pd.read_csv('/share/home/konglingcong/vegi/doc_data/table/tb_CMD.csv', index_col=0)
        self.data = []
        self.length = []

        for i in tb.file:
            img = nib.load(i)
            img = np.transpose(np.asanyarray(img.dataobj), (3, 0, 1, 2))
            img = img
            img = torch.tensor(img, dtype=torch.float32)
            self.data.append(img)
            self.length.append(img.shape[0])
        self.labels = tb.type
        self.labels = torch.tensor(self.labels, dtype=torch.int8)
        print(self.length)
        # self.data = torch.concat(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
