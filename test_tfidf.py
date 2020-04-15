import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

clf = torch.load('models/logistic.pt')
vectorizer1 = torch.load('models/vectorizer1.pt')
vectorizer2 = torch.load('models/vectorizer2.pt')

data_test=pd.read_csv("data/snli_1.0/snli_1.0_test.csv",header=0)
array_test=data_test.as_matrix(('sentence1','sentence2'))
y_test=data_test.as_matrix(('label',))


x1=vectorizer1.transform(array_test[:,0])
x2=vectorizer2.transform(array_test[:,1])
x=hstack([x1,x2])

y_out=clf.predict(x)

y_t=y_test[:,0]
y_t[y_t=='neutral']=0
y_t[y_t=='contradiction']=1
y_t[y_t=='entailment']=2


print('accuracy: ',np.mean(y_out == y_t))


def write_file(filename,y):
    with open(filename, 'w') as f:
            for i in range(len(y)):
                f.write("{}\n".format(key(y[i])))

def key(a):
    if a==2:
        return 'entailment'
    elif a==1:
        return 'contradiction'
    else :
        return 'neutral'


write_file('tfidf.txt',y_out)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
y1=np.array(y_t,dtype='int')
y2=np.array(y_out,dtype='int')
labels=[0,1,2]
cm1=confusion_matrix(y1,y2,labels)
df_cm = pd.DataFrame(cm1, index = ['neutral','contradiction','entailment'],columns=['neutral','contradiction','entailment'])
plt.figure(figsize = (8,5.5))
sn.heatmap(df_cm, annot=True)
plt.title("Confusion Matrix")
plt.savefig('tf_idf confusion')
