#d00居然是反的？？？

import pandas as pd
#有么有改进方法？
#初步定为：4个train(0-3)，后三个不在train中出现，为异常检测部分
#从哪里学习数据处理的操作？
f0=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d00.dat',encoding='utf-8')
f1=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d01.dat',encoding='utf-8')
f2=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d02.dat',encoding='utf-8')
f3=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d03.dat',encoding='utf-8')
f4=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d04.dat',encoding='utf-8')
f5=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d05.dat',encoding='utf-8')
f6=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d06.dat',encoding='utf-8')
f7=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d07.dat',encoding='utf-8')
f8=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d08.dat',encoding='utf-8')
f9=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d09.dat',encoding='utf-8')
f10=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d10.dat',encoding='utf-8')
f11=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d11.dat',encoding='utf-8')
f12=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d12.dat',encoding='utf-8')
f13=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d13.dat',encoding='utf-8')
f14=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d14.dat',encoding='utf-8')
f15=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d15.dat',encoding='utf-8')
f16=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d16.dat',encoding='utf-8')
f17=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d17.dat',encoding='utf-8')
f18=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d18.dat',encoding='utf-8')
f19=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d19.dat',encoding='utf-8')
f20=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d20.dat',encoding='utf-8')
f21=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d21.dat',encoding='utf-8')
trains = []
trains_label = []
#0
sentimentlists = []
for line in f0:
    s = line.strip().split('\t')
    sentimentlists.append(s)
f0.close()

trains0 = []
trains0_label =[]
for i in range(len(sentimentlists)):
    train0 = sentimentlists[i][0].split()#sentimentlist 是一个2维列表，第二维只有一个元素，故要加上0
    for j in range(len(train0)):
        train0[j] = float(train0[j])
    trains0.append(train0)
    trains0_label.append(0)
#由于d00.dat出现了问题，在此手动对其转置
trains0_t = []
for i in range(len(trains0[0])):
    trains0_t1 = []
    for j in range(len(trains0)):
        trains0_t1.append(trains0[j][i])
    trains0_t.append(trains0_t1)
    pass
for i in range(len(trains0_label),len(trains0_t)):
    trains0_label.append(0)
    pass

trains.append(trains0_t)
trains_label.append(trains0_label)


#1
sentimentlists = []
for line in f1:
    s = line.strip().split('\t')
    sentimentlists.append(s)
f1.close()

trains1 = []
trains1_label =[]
for i in range(len(sentimentlists)):
    train1 = sentimentlists[i][0].split()#sentimentlist 是一个2维列表，第二维只有一个元素，故要加上0
    for j in range(len(train1)):
        train1[j] = float(train1[j])
    trains1.append(train1)
    trains1_label.append(1)
trains.append(trains1)
trains_label.append(trains1_label)

#2
sentimentlists = []
for line in f2:
    s = line.strip().split('\t')
    sentimentlists.append(s)
f2.close()

trains2 = []
trains2_label =[]
for i in range(len(sentimentlists)):
    train2 = sentimentlists[i][0].split()#sentimentlist 是一个2维列表，第二维只有一个元素，故要加上0
    for j in range(len(train2)):
        train2[j] = float(train2[j])
    trains2.append(train1)
    trains2_label.append(2)
trains.append(trains2)
trains_label.append(trains2_label)

#3
sentimentlists = []
for line in f3:
    s = line.strip().split('\t')
    sentimentlists.append(s)
f3.close()

trains3 = []
trains3_label =[]
for i in range(len(sentimentlists)):
    train3 = sentimentlists[i][0].split()#sentimentlist 是一个2维列表，第二维只有一个元素，故要加上0
    for j in range(len(train3)):
        train3[j] = float(train3[j])
    trains3.append(train1)
    trains3_label.append(3)
trains.append(trains3)
trains_label.append(trains3_label)


# df_train=pd.DataFrame(sentimentlists)
#.00和后面的列数居然对不上？
print(len(trains))
print(len(trains[0]))
print(len(trains[0][0]))
print(len(trains_label))
print(len(trains_label[0]))
# print(len(trains_label[0][0]))

import torch 
from torch.utils.data import Dataset

class TEDataset(Dataset):
    #有可能一次处理一个trains[i]
    def __init__(self, X, y=None):
        temp = []
        for i in range(len(X)):
            for j in range(len(X[i])):
                temp.append(X[i][j])
            pass
        pass
        self.data = torch.tensor(temp)
        
        print(len(self.data))
        if y is not None:
            temp = []
            for i in range(len(y)):
                for j in range(len(y[i])):
                    temp.append(y[i][j])
                pass
            pass
            self.label=torch.tensor(temp)
            print(self.label)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

import numpy as np


VAL_RATIO = 0.25

trains_np = np.array(trains)
trains_np_label = np.array(trains_label)

percent = int(len(trains) * (1 - VAL_RATIO))

train_x, train_y, val_x, val_y = trains[:percent], trains_label[:percent], trains[percent:], trains_label[percent:]

print(len(train_x))
print(len(train_x[0]))
print(len(train_x[0][0]))
print(len(train_y))
print(len(train_y[0]))
# print(len(train_y[0][0]))

BATCH_SIZE = 64

from torch.utils.data import DataLoader

train_set = TEDataset(train_x, train_y)
val_set = TEDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

import torch
import torch.nn as nn
#输入有52个向量，输出目前有四个，信息分数可以根据四个输出进行判断
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.layer1 = nn.Linear(429, 1024)
        # self.layer2 = nn.Linear(1024, 512)
        # self.layer3 = nn.Linear(512, 128)
        # self.out = nn.Linear(128, 39) 

        # self.act_fn = nn.Sigmoid()
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(52, 200),
            #nn.Sigmoid(),
            nn.LeakyReLU(0.02),

            # nn.LayerNorm(200),
            
            nn.Linear(200, 4),
            nn.Sigmoid(),
            # nn.Softmax()
            #nn.LeakyReLU(0.02)
        )

    def forward(self, x):
        return self.model(x)
# start training
#问题：交叉熵不要自己独热化，给0到n-1作为标签即可

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Feel free to change the training parameters here.
# fix random seed for reproducibility
same_seeds(0)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 20               # number of training epoch
learning_rate = 0.1       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#可能是因为没有初始化？
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
model.apply(init_weights)

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs),print(labels)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        print(f'Train:{outputs}')
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        # print(f'train:{train_pred}')
        batch_loss.backward() 
        optimizer.step() 
        
        print(train_pred.cpu())
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')