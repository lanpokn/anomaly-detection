#Preparing data
import pandas as pd

f=open(r'/home/lanpokn/Documents/vscode/python project/srtp/anomaly detect/TE REAL/d00.dat',encoding='utf-8')

sentimentlists = []
for line in f:
    s = line.strip().split('\t')
    print(s)
    sentimentlists.append(s)
f.close()

trains = []
for line in sentimentlists:
    train = line.split()
    trains.append(train)

df_train=pd.DataFrame(sentimentlists)
print(format(df_train.shape))