import numpy as np
import pandas as pd

from google.colab import files
uploaded =files.upload()

dataset=pd.read_csv("ml1.csv")

print(dataset)

dataset.head()

dataset.shape

dataset.tail()

x=dataset.iloc[:-1,:-1].values

y=dataset.iloc[:-1,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

dataset.isnull().any()

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix : ")
print(cm)
print("Accuracy of the model:{0}%".format(accuracy_score(y_test,y_pred)*100))

age=int(input("Enter new customer age:"))
sal=int(input("enter new customer salary"))
newcust = [[age,sal]]
result=model.predict(sc.transform(newcust))
print(result)
if result==1:
  print("customer will buy")
else:
  print("customer wont buy")