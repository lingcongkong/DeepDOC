import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from glob import glob
import numpy as np
import cv2

def find_nii(data_path, is_func=False):
    if is_func:
        fa = 'f'
    else:
        fa = 'a'
    for p in os.listdir(data_path):
        for nii in os.listdir(os.path.join(data_path, p, 'preprocessed')):
            if nii[4] == fa:
                yield int(p[1:]), os.path.join(data_path, p, 'preprocessed', nii)


class get_dataset(Dataset):
    def __init__(self, data_path, stage):
        self.stage = stage
        self.func_data = []
        self.idx = []
        self.label_doc = []
        self.label_awareness = []
        for pid, i in find_nii(data_path, is_func=True):
            img = nib.load(i)
            img = np.transpose(np.asanyarray(img.dataobj), (3, 0, 1, 2))
            img = img[0:236]
            f = torch.tensor(img, dtype=torch.float32)
            self.func_data.append(f)
            self.idx += [pid for j in range(f.shape[0])]
        self.func_data = torch.cat(self.func_data, 0)
        for i in self.idx:
            if i < 52:
                self.label_doc.append(1)
            else:
                self.label_doc.append(0)
            if i < 38:
                self.label_awareness.append(1)
            else:
                self.label_awareness.append(0)
        self.label_awareness = torch.tensor(self.label_awareness, dtype=torch.long)
        self.label_doc = torch.tensor(self.label_doc, dtype=torch.long)
    def __len__(self):
        return self.func_data.shape[0] if self.stage == 1 else 51


    def __getitem__(self, x):
        return self.func_data[x], self.label_awareness[x] if self.stage == 2 else self.label_doc[x], self.idx[x]


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix