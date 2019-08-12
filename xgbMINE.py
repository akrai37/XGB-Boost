import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Churn_Modelling.csv')
x= dataset.iloc[:, 3:13].values
y= dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
x[:, 1]= labelencoder_X_1.fit_transform(x[:, 1])
labelencoder_X_2= LabelEncoder()
x[:, 2]= labelencoder_X_2.fit_transform(x[:, 2])
onehotencoder= OneHotEncoder(categorical_features = [1])
x= onehotencoder.fit_transform(x).toarray() 
x= x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state= 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier= XGBClassifier()
classifier.fit(x_train, y_train) 

y_pred= classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred, y_test)

from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator= classifier , X=x_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()





















