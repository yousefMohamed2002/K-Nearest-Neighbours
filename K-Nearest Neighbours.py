import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  metrics
path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
df=pd.read_csv(path)
print(df.columns)
x=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
y=df[["custcat"]].values
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
k=4
K_Model =KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
yhat=K_Model.predict(X_test)
# print("predict",yhat[0:5])
# print("real",y_test[0:5])
print ("accuracy of Model",metrics.accuracy_score(y_test,yhat) )

