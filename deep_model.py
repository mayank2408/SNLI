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
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=Batch_Size, device=device)


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


def train(model,train_loader,val_loader,optimizer,criterion,scheduler,epochs,print_iter=5):
    train_loss=[]
    val_loss=[]
    for i in range(epochs):
        model.train()
        train_loader.init_epoch()
        running_loss_train=0 
        total=0.0
        for indx,inputs in enumerate(train_loader):
            #inputs=inputs.to(device)
            #labels=labels.to(device)
            optimizer.zero_grad()
            output=model(inputs)
            loss=criterion(output,inputs.label)
            running_loss_train+=loss.item()
            loss.backward()
            optimizer.step()
            total+=inputs.batch_size
        train_loss.append(running_loss_train/total)
        if (i%print_iter)==0:
            model.eval()
            running_corrects=0.0
            running_loss=0.0
            total=0.0
            with torch.no_grad():
                for inputs in val_loader:
                    #inputs=inputs.to(device)
                    #labels=labels.to(device)
                    optimizer.zero_grad()
                    output=model(inputs)
                    loss=criterion(output,inputs.label)
                    _,pred=torch.max(output, 1)
                    running_corrects += torch.sum(pred == inputs.label).item()
                    running_loss+=loss.item()
                    total+=inputs.batch_size
            print(' {} Loss: {:.6f} Acc: {:.6f}'.format(
                  i,running_loss/total,(running_corrects/total)))
            val_loss.append(running_loss/total)
        scheduler.step()
    return model,train_loss,val_loss


model2=Classifier()


import torch.optim as optim
lr=0.005
optimizer2=optim.Adam(model2.parameters(),lr,weight_decay=0.0001)
criterion2=nn.CrossEntropyLoss()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.55)
model2.to(device)


_,train_loss,val_loss=train(model2,train_iter,dev_iter,optimizer2,criterion2,exp_lr_scheduler2,epochs=35,print_iter=1)


def accuracy(model,train_loader):
    model.eval()
    running_corrects=0.0
    running_loss=0.0
    total=0.0
    with torch.no_grad():
        for inputs in train_loader:
            #inputs=inputs.to(device)
            #labels=labels.to(device)
            output=model(inputs)
            _,pred=torch.max(output, 1)
            running_corrects += torch.sum(pred == inputs.label)
            total+=inputs.batch_size
    print(' Acc: {:.6f}'.format((running_corrects/total)))
    return running_corrects/total


accuracy(model2,test_iter)


torch.save(model2.state_dict(), 'models/hs256.pt')


from matplotlib import pyplot as plt
plt.figure()
plt.plot(train_loss,label='train_loss')
plt.plot(val_loss,label='validation_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper left')
plt.savefig("./loss.jpg")

torch.save(sentences,'models/vocab')