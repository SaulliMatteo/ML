import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

ds = pd.read_csv('ML/tutorial/sonar_data.csv', header=None)

x = ds.drop(columns=60, axis=1)
y = ds[60]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3, stratify=y)

model = LogisticRegression()
model.fit(x_train, y_train)

testing_pred = model.predict(x_train)
training_data_pred = accuracy_score(testing_pred, y_train)

testing_pred = model.predict(x_test)
testing_data_pred = accuracy_score(testing_pred, y_test)

data = np.asarray(str(input("inserisci i dati: ")).split(','), dtype=float)
data = data.reshape(1, -1)
pred = model.predict(data)
os.system('cls')

if(pred[0] == 'R'):
    print("e' una roccia")
else:
    print("e' una mina")


