import pandas as pd
import numpy as np

titanic = pd.read_excel(r"titanic_streamlit.xlsx")
df = titanic.copy()

target = 'survived'
encode = ['sex','embarked']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Separating X and y
X = df.drop('survived', axis=1)
Y = df['survived']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open(r'titanic_clf_2.pkl', 'wb'))
