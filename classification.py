#data load
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import minmax_scale,StandardScaler

#1D target tensor expected, multi-target not supported,pytorch再这一点上与sklearn很不一样，建议中期报告说这个事情
def creat_dataset(test_index = [0, 1, 2]):
    path = './TE_mat_data/'
    print("loading data...")

    fault1 = loadmat(path + 'd01.mat')['data']
    fault2 = loadmat(path + 'd02.mat')['data']
    fault3 = loadmat(path + 'd03.mat')['data']
    fault4 = loadmat(path + 'd04.mat')['data']
    fault5 = loadmat(path + 'd05.mat')['data']
    fault6 = loadmat(path + 'd06.mat')['data']
    fault7 = loadmat(path + 'd07.mat')['data']
    fault8 = loadmat(path + 'd08.mat')['data']
    fault9 = loadmat(path + 'd09.mat')['data']
    fault10 = loadmat(path + 'd10.mat')['data']
    fault11 = loadmat(path + 'd11.mat')['data']
    fault12 = loadmat(path + 'd12.mat')['data']
    fault13 = loadmat(path + 'd13.mat')['data']
    fault14 = loadmat(path + 'd14.mat')['data']
    fault15 = loadmat(path + 'd15.mat')['data']

    attribute_matrix_ = pd.read_excel('./attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values

    #这一步要自己手调
    # train_index = list(set(np.arange(15)) - set(test_index))
    train_index = test_index[:][:]

    test_index.sort()
    train_index.sort()

    print("test classes: {}".format(test_index))
    print("train classes: {}".format(train_index))

    data_list = [fault1, fault2, fault3, fault4, fault5,
                 fault6, fault7, fault8, fault9, fault10,
                 fault11, fault12, fault13, fault14, fault15]

    trainlabel = []
    train_attributelabel = []
    traindata = []
    for item in train_index:
        trainlabel += [item] * 480
        train_attributelabel += [attribute_matrix[item, :]] * 480
        traindata.append(data_list[item])
    trainlabel = np.row_stack(trainlabel)
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.column_stack(traindata).T

    testlabel = []
    test_attributelabel = []
    testdata = []
    for item in test_index:
        testlabel += [item] * 480
        test_attributelabel += [attribute_matrix[item, :]] * 480
        testdata.append(data_list[item])
    testlabel = np.row_stack(testlabel)
    test_attributelabel = np.row_stack(test_attributelabel)
    testdata = np.column_stack(testdata).T

    return traindata, trainlabel, train_attributelabel, \
           testdata, testlabel, test_attributelabel, \
           attribute_matrix_.iloc[test_index,:], attribute_matrix_.iloc[train_index, :]

print("==========================[train classes][0 ,1, 2]===================================")
print("beginning...with feature extraction")
traindata, trainlabel, train_attributelabel, testdata, testlabel, \
test_attributelabel, attribute_matrix, train_attribute_matrix = creat_dataset([0, 1, 2])
#USE D21 AS UNKONOWN
#PREprocess,先minmax再standardscaler
#1D target tensor expected, multi-target not supported,pytorch再这一点上与sklearn很不一样，建议中期报告说这个事情
traindata = minmax_scale(traindata)
testdata = minmax_scale(testdata)
percent = 0.8
partition = int(traindata.shape[0]*percent)
train_x, train_y, val_x, val_y = traindata[:partition], trainlabel[:partition], traindata[partition:], trainlabel[partition:]
train_x[0]

#creat dataset
import torch 
from torch.utils.data import Dataset

class TEDataset(Dataset):
    #有可能一次处理一个trains[i]
    def __init__(self, X, y=None):
        self.data = torch.tensor(X)
        
        if y is not None:
            self.label=torch.tensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

#dataset and dataloader

BATCH_SIZE = 64

from torch.utils.data import DataLoader

train_set = TEDataset(train_x, train_y)
val_set = TEDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

#create nerual network

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
            nn.Linear(52, 105),#2*输入加1
            #nn.Sigmoid(),
            nn.LeakyReLU(0.02),

            # nn.LayerNorm(200),
            
            nn.Linear(105, 3),
            nn.Sigmoid(),
            # nn.Softmax()
            #nn.LeakyReLU(0.02)
        )

    def forward(self, x):
        return self.model(x)

#check device
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

# start training
#问题：交叉熵不要自己独热化，给0到n-1作为标签即可
# 为了解决：expected scalar type Float but found Double
model = model.double()
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