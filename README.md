# Implementation-of-SVM-For-Spam-Mail-Detection
AIM:
To write a program to implement the SVM For Spam Mail Detection.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Import the required packages.
Import the dataset to operate on.
Split the dataset.
Predict the required output.
End the program.
Program:
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MS.JEEVA
RegisterNumber:  212221230040
*/

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


Output:
SVM For Spam Mail Detection

SVM For Spam Mail Detection

SVM For Spam Mail Detection

SVM For Spam Mail Detection

SVM For Spam Mail Detection

SVM For Spam Mail Detection

SVM For Spam Mail Detection

Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming
