# Implementation-of-SVM-For-Spam-Mail-Detection
### NAME : HASHWATHA M
### REG NO : 212223240051
## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HASHWATHA M
RegisterNumber: 212223240051
*/

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
#countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
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

from sklearn.metrics import confusion_matrix,classification_report
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:

## data.head:

![image](https://github.com/user-attachments/assets/2db24cd6-d703-48d8-89fd-f3b679568d54)

## data.info:

![image](https://github.com/user-attachments/assets/a76433f6-a1d9-4837-9ff5-8ebd71240a31)

## data.isnull:

![image](https://github.com/user-attachments/assets/3fe9c8b1-c9c0-4405-a1da-f88d155f81b9)

## y_predict :

![image](https://github.com/user-attachments/assets/e64fe523-8aa2-4427-9e6e-c53caf41d2df)

## accuracy :

![image](https://github.com/user-attachments/assets/ce07e77e-88c5-4c46-8a43-a570cff3c18a)

## confusion matrix:

![image](https://github.com/user-attachments/assets/367672fc-fc3b-4a09-8e54-c4a4720a8ac9)

## classification_report :

![image](https://github.com/user-attachments/assets/5138fabc-bba4-4cbf-9b51-883f30707da2)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
