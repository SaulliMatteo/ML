import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mail_data = pd.read_csv('ML/tutorial/mail_data.csv')

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

x = mail_data['Message']
y = mail_data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase = True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_freatures = feature_extraction.transform(x_test)

y_train, y_test = y_train.astype('int'), y_test.astype('int')

model = LogisticRegression()
model.fit(x_train_features, y_train)

user_input = input('inserisci un testo: ')
user_input = np.asarray([user_input])
user_data = feature_extraction.transform(user_input)
model_pred = model.predict(user_data)
model_acc = model.predict_proba(user_data)

if(model_pred == 0):
    print(f"e' uno spam al {model_acc[0][0]*100:.2f}%")

else:
    print(f"non e' uno spam al {model_acc[0][1]*100:.2f}%")

