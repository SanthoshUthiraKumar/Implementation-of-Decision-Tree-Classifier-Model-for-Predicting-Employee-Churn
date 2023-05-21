# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Libraries.
 
2.Upload the dataset in the compiler and read the dataset.

3.Find head,info and null elements in the dataset.

4.Using LabelEncoder and DecisionTreeClassifier , find accuracy and prediction for the dataset.

5.End the program.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Santhosh U
RegisterNumber:  212222240092
*/

import pandas as pd
data=pd.read_csv("/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### 1. data.head()
![Output1](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/4c274c88-b0ea-49b1-83d3-bf8e45b842df)

### 2. data.info()
![Output2](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/214cac1b-e052-4aa9-ac1d-e5b7763bc4fe)

### 3. isnull() and sum()
![Output3](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/665a0cc3-18df-40bf-b36d-df4dc95c7af9)

### 4. data value counts()
![Output4](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/720074fe-610a-47b4-a328-0b218e6ae37d)

### 5. data.head() for salary
![Output5](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/01505622-1d26-4244-964a-dba62abe45cb)

### 6. x.head()
![Output6](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/ef93e2fe-c20c-4207-96ff-5044eee568dd)

### 7. accuracy value
![Output7](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/85c3b8e3-fb43-43ee-a21d-e9bf95b741f4)

### 8. data prediction
![Output8](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477975/c9bda6fc-cd97-4f8f-83da-4613fbb04df3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
