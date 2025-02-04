import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from os import system

ds = pd.read_csv('ML/tutorial/diabetes (2).csv')

x = ds.drop(columns='Outcome', axis=1)
y = ds['Outcome']

scalar = StandardScaler()
x = scalar.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=5)

model = LogisticRegression()
model.fit(x_train, y_train)
testing_data_pred = model.predict(x_test)

data = np.asarray(str(input("inserisci i dati: ")).split(','), dtype=float)
data = data.reshape(1, -1)
data = scalar.transform(data)
pred = model.predict(data)
proba = model.predict_proba(data)[0]

system('cls')
print(proba)
if(pred[0]):
    print('diabetic')
else:
    print('non diabetic')