# Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
5.End the program.

## Program:
~~~
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Guhanandan V
RegisterNumber:  212221220014
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
~~~

## Output:

![op1](https://user-images.githubusercontent.com/100425381/204842136-9037ce58-1681-4823-995c-47716df9c9d6.png)

![op2](https://user-images.githubusercontent.com/100425381/204842154-c52e3ebc-75fd-4cb3-aac8-8bdf605352d6.png)

![op3](https://user-images.githubusercontent.com/100425381/204842167-e1cc768c-904b-4fb1-a9c8-489f54bc1534.png)

![op4](https://user-images.githubusercontent.com/100425381/204842180-6de43d27-c378-455b-9534-88c846864f3e.png)

![op5](https://user-images.githubusercontent.com/100425381/204842194-adf42835-0f5a-44d6-805d-9a54cb732677.png)

![op6](https://user-images.githubusercontent.com/100425381/204842208-a0cd8e51-9144-4545-abc7-0dd709a9dd87.png)

![op7](https://user-images.githubusercontent.com/100425381/204842265-79682cd9-b751-41f4-9d20-73c7c312214b.png)


## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming
