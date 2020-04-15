import pandas as pd
import numpy as np
import torch
import torchtext
from torchtext.data import Field, BucketIterator
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets

import spacy
spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

sentences = data.Field(lower=True, tokenize=tokenizer)
ans = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(sentences, ans)

sentences.build_vocab(train, dev, test,min_freq=3)
ans.build_vocab(train, dev, test)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
Batch_Size=128
test_iter= data.BucketIterator(test,batch_size=Batch_Size,shuffle=False)


n_layer=1
class My_RNN(nn.Module):

    def __init__(self, embed_dim,hidden_dim,drop_p):
        super(My_RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                        num_layers=n_layer, dropout=drop_p,bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = 2*n_layer, batch_size, hidden_dim
        h0 = c0 = inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
    
    

class Output(nn.Module):

    def __init__(self, out_dim,inp_dim,drop_p):
        super(Output, self).__init__()
        self.fc1=nn.Linear(inp_dim,int(inp_dim/2))
        self.fc2=nn.Linear(int(inp_dim/2),int(inp_dim/2))
        self.fc3=nn.Linear(int(inp_dim/2),int(inp_dim/4))
        self.fc4=nn.Linear(int(inp_dim/4),out_dim)
        self.p=drop_p

    def forward(self, x):
        x=F.dropout(F.relu(self.fc1(x)),p=self.p)
        x=F.dropout(F.relu(self.fc2(x)),p=self.p)
        x=F.dropout(F.relu(self.fc3(x)),p=self.p)
        x=(self.fc4(x))
        return x

hidden_dim=256
embed_dim=300
out_dim=4
drop_p1=0.25
drop_p2=0.3
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.embedding=nn.Embedding(len(sentences.vocab),embed_dim)
        self.RNN=My_RNN(embed_dim,hidden_dim,drop_p1)
        self.final_l=Output(out_dim,4*hidden_dim,drop_p2)
        
    def forward(self,batch):
        sen1 = self.embedding(batch.premise)
        sen2 = self.embedding(batch.hypothesis)
        premise = self.RNN(sen1)
        hypothesis = self.RNN(sen2)
        out = self.final_l(torch.cat([premise, hypothesis], 1))
        return out


model2=Classifier()

model2.load_state_dict(torch.load('models/hs256.pt',device))

def accuracy(model,train_loader):
    model.eval()
    running_corrects=0.0
    running_loss=0.0
    total=0.0
    with torch.no_grad():
        for inputs in train_loader:
            #inputs=inputs.to(device)
            #labels=labels.to(device)
            #print(inputs.label)
            output=model(inputs)
            _,pred=torch.max(output, 1)
            running_corrects += torch.sum(pred == inputs.label)
            total+=len(inputs.label)
    print(' Acc: {:.6f}'.format((running_corrects/total)))
    return running_corrects/total

accuracy(model2,test_iter)

test_iter= data.BucketIterator(test,batch_size=1,shuffle=False)
def write_file(filename,test_iter,model):
    with open(filename, 'w') as f:
        model.eval()
        with torch.no_grad():
            for inputs in test_iter:
                #inputs=inputs.to(device)
                #labels=labels.to(device)
                #print(inputs.label)
                output=model(inputs)
                _,pred=torch.max(output, 1)
                f.write("{}\n".format(key(pred)))

def key(a):
    if a==1:
        return 'entailment'
    elif a==2:
        return 'contradiction'
    else :
        return 'neutral'

write_file('deep_model.txt',test_iter,model2)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y1=np.array([],dtype='int')
y2=np.array([],dtype='int')
model2.eval()
with torch.no_grad():
    for inputs in test_iter:
        output=model2(inputs)
        _,pred=torch.max(output, 1)
        y2=np.append(y2,pred)
        y1=np.append(y1,inputs.label)

labels=[3,2,1]
cm1=confusion_matrix(y1,y2,labels)
df_cm = pd.DataFrame(cm1, index = ['neutral','contradiction','entailment'],columns=['neutral','contradiction','entailment'])
plt.figure(figsize = (8,5.5))
sn.heatmap(df_cm, annot=True)
plt.title("Confusion Matrix")
plt.savefig('deep_confusion')
