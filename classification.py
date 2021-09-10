#Preparing data
import pandas as pd
import numpy as np

f=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d00.dat',encoding='utf-8')

sentimentlists = []
for line in f:
    s = line.strip().split('\t')
    sentimentlists.append(s)
f.close()

#各种声明
trains0 = []
for i in range(len(sentimentlists)):
    train = sentimentlists[i][0].split()
    trains0.append(train)
# df_train=pd.DataFrame(sentimentlists)
print(len(trains0[0]))

import torch 
from torch.utils.data import Dataset

class TEDataset(Dataset):
    #X:trains,y:labels
    def __init__(self, X, y=None):
        self.data = torch.tensor(X)
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)
