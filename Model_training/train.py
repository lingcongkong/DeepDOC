from efficientnet_pytorch_3d import EfficientNet3D
import torch
from torchsummary import summary
from dataset import CustomDataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
import os
from utils import confusion_matrix
from sklearn.metrics import roc_auc_score

DEVICE = 'cuda'
EPOCH = 200
BATCHSIZE = 128
SEED = 42
B = 3
MODEL_NAME = "efficientnet-b{}".format(B)
LR = 5e-4
MT=0.009
FOLD = 5
SAVEPATH = '~/'

def savemodel(model, auc, epoch, fold, save_model_path=SAVEPATH):
    with open(os.path.join(save_model_path, f'model_f{fold}_ep{epoch}_{auc:.4f}.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)


def expand_labels(label, l):
    labels = torch.empty((np.sum(l)), dtype=torch.int64)
    t = 0
    for i in range(label.shape[0]):
        labels[t:t+l[i]] = torch.repeat_interleave(label[i], l[i])
        t += l[i]
    return labels
    
class CD(Dataset):
    def __init__(self, img, labels):
        self.img = img
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.img[idx], self.labels[idx] 

def check_path(in_path):
    if not os.path.exists(in_path):
        os.mkdir(in_path)
        
        
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  

def train(model, train_loader, optimizer, criterion, DEVICE):
    epoch_loss = 0
    cm = torch.zeros((2, 2))
    y_true, y_score = [], []
    for data, labels in train_loader:
        data = torch.unsqueeze(data, 1)
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        output = model(data)
        loss = criterion(output, labels)

        pred = output.argmax(dim=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(labels, output)
        y_true.append(labels.cpu().detach())
        y_score.append(output.cpu().detach().softmax(0))

        epoch_loss += loss.cpu().detach().item()
        cm = confusion_matrix(pred.cpu().detach(), labels.cpu().detach(), cm)
        
    y_true = torch.concat(y_true)
    y_score = torch.concat(y_score)[:, 1]

    acc = cm.diag().sum() / cm.sum()
    spe, sen = cm.diag() / (cm.sum(dim=0) + 1e-6)
    pre = cm.diag()[1] / (cm.sum(dim=1) + 1e-6)[1]
    rec = sen
    f1score = 2*pre*rec / (pre+rec+ 1e-6)
    auc = roc_auc_score(y_true, y_score)
    avg_epoch_loss = epoch_loss / len(train_loader)
    return [avg_epoch_loss, acc, sen, spe, auc, pre, f1score]

def val(model, val_loader, criterion, DEVICE):
    epoch_loss = 0
    cm = torch.zeros((2, 2))
    y_true, y_score = [], []
    for data, labels in val_loader:
        data = torch.unsqueeze(data, 1)
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        output = model(data)
        loss = criterion(output, labels)
        pred = output.argmax(dim=1)

        y_true.append(labels.cpu().detach())
        y_score.append(output.cpu().detach().softmax(0))

        epoch_loss += loss.cpu().detach().item()
        cm = confusion_matrix(pred.cpu().detach(), labels.cpu().detach(), cm)
    
    y_true = torch.concat(y_true)
    y_score = torch.concat(y_score)[:, 1]    
    acc = cm.diag().sum() / cm.sum()
    spe, sen = cm.diag() / (cm.sum(dim=0) + 1e-6)
    pre = cm.diag()[1] / (cm.sum(dim=1) + 1e-6)[1]
    rec = sen
    f1score = 2*pre*rec / (pre+rec+ 1e-6)
    auc = roc_auc_score(y_true, y_score)
    avg_epoch_loss = epoch_loss / len(val_loader)
    return [avg_epoch_loss, acc, sen, spe, auc, pre, f1score]


if __name__ == '__main__':
    seed_everything(SEED)
    writer = pd.DataFrame(columns=['fold', 'epoch', 'phase', 'loss', 'acc',
     'sen', 'spe', 'auc', 'pre', 'f1score'])
    num_classes = 2

    criterion = torch.nn.CrossEntropyLoss()

    # 创建数据集
    dataset = CustomDataset()

    # StratifiedKFold 分层抽样
    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)
    tb = pd.read_csv('tb_MCSVS.csv', index_col=0)
    data = tb.file
    labels = tb.type


    for fold, (train_idx, val_idx) in enumerate(skf.split(
    np.linspace(0, len(dataset)-1, len(dataset)), 
    tb.type)):
    
        model = EfficientNet3D.from_name(MODEL_NAME,
        override_params={'num_classes': 2, 'image_size':73}, in_channels=1)
        msd = torch.load('./models/en3d_b{}.pth'.format(B))
        model.load_state_dict(msd)
        model = model.to(DEVICE)
        num_classes = 2

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MT)


        # 创建 Dataset
        # 创建 Dataset
        train_data = []
        train_label = []
        train_len = []
        for i in train_idx:

            if dataset[i][1] == 0:
                for j in range(3):
                    train_data.append(dataset[i][0])
                    train_label.append(dataset[i][1])
                    train_len.append(dataset.length[i])
            else:
                train_data.append(dataset[i][0])
                train_label.append(dataset[i][1])
                train_len.append(dataset.length[i])
        train_data = torch.concat(train_data)
        train_label = torch.tensor(train_label, dtype=torch.int8)

        val_data = []
        val_label = []
        val_len = []
        for i in val_idx:
            val_data.append(dataset[i][0])
            val_label.append(dataset[i][1])
            val_len.append(dataset.length[i])
        val_data = torch.concat(val_data)
        val_label = torch.tensor(val_label, dtype=torch.int8)

        # expand and shuffle the subset
        

        train_label = expand_labels(train_label, train_len)
        val_label = expand_labels(val_label, val_len)

        train_indices = torch.randperm(train_label.shape[0])
        train_data = train_data[train_indices]
        train_label = train_label[train_indices]

        val_indices = torch.randperm(val_label.shape[0])
        val_data = val_data[val_indices]
        val_label = val_label[val_indices]     

        # 创建 DataLoader
        train_subset = CD(train_data, train_label)
        val_subset = CD(val_data, val_label)

        train_loader = DataLoader(train_subset, batch_size=BATCHSIZE, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=BATCHSIZE, shuffle=False)

        val_b_auc = 0
        for epoch in range(EPOCH):
    
            
            model.train()
            # 训练和验证过程
            avg_epoch_loss, acc, sen, spe, auc, pre, f1score = \
            train(model, train_loader, optimizer, criterion, DEVICE)
            writer = pd.concat((writer, pd.DataFrame(
            [[fold, epoch, 'train', avg_epoch_loss, acc, sen, spe, auc, pre, f1score]],
            columns=['fold', 'epoch','phase', 'loss', 'acc',
            'sen', 'spe', 'auc', 'pre', 'f1score'])))
            
            # 验证模型
            model.eval()
            avg_epoch_loss, acc, sen, spe, auc, pre, f1score = \
            val(model, val_loader, criterion, DEVICE)

            # save model
            if auc > val_b_auc:
                val_b_auc = auc
                savemodel(model, auc, epoch, fold)
            
            writer = pd.concat((writer, pd.DataFrame(
            [[fold, epoch, 'val', avg_epoch_loss, acc, sen, spe, auc, pre, f1score]],
            columns=['fold', 'epoch','phase', 'loss', 'acc',
            'sen', 'spe', 'auc', 'pre', 'f1score'])))
            # print(fold) 
            
            writer.to_csv(f'{SAVEPATH}/{MODEL_NAME}_{SEED}.csv')










