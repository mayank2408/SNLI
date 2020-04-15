import pandas as pd
import numpy as np


data=pd.read_csv("data/snli_1.0/snli_1.0_train.csv",header=0)

numpy_array1 = data[data.label=='neutral'].as_matrix()[:,:-1]
y1=np.zeros((numpy_array1.shape[0],1))
numpy_array2 = data[data.label=='contradiction'].as_matrix()[:,:-1]
y2=np.ones((numpy_array2.shape[0],1))
numpy_array3 = data[data.label=='entailment'].as_matrix()[:,:-1]
y3=2*np.ones((numpy_array3.shape[0],1))

numpy_array=np.append(numpy_array1,numpy_array2,axis=0)
numpy_array=np.append(numpy_array,numpy_array3,axis=0)
y=np.append(y1,y2,axis=0)
y=np.append(y,y3,axis=0)
#print(numpy_array.shape)
#print(y.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer1 = TfidfVectorizer(max_df=0.99,
                        min_df=5,use_idf=True,
                        ngram_range=(1, 2))
X1=vectorizer1.fit_transform(numpy_array[:,0])
vectorizer2 = TfidfVectorizer(max_df=0.99,
                        min_df=5,use_idf=True,
                        ngram_range=(1, 2))
X2=vectorizer2.fit_transform(numpy_array[:,1])

from scipy.sparse import hstack
X=hstack([X1,X2])


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X, y)


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

print('accuracy',np.mean(y_out == y_t))


import torch
torch.save(clf, 'models/logistic.pt')

torch.save(vectorizer1, 'models/vectorizer1.pt')

torch.save(vectorizer2, 'models/vectorizer2.pt')
