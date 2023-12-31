# -*- coding: utf-8 -*-
"""Datahackathon v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mBicZMLvYqKkRglI01Lh09dB0Y56wM-A
"""

!pip install lime

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import lime
from lime import lime_tabular

#inputs
readFilePath = "TransformedData.csv"

dataframe = pd.read_csv(readFilePath)

features = ["trans_type",
            "amount",
            "trans_nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "trans_nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFlaggedFraud"]

predict = "isFraud"

test_size=0.2
random_state=42

#PredictFeatures = [4,181,3101, 181, 0, 8961, 0, 0,  0]
PredictFeatures = [4,181,3101, 181, 0, 8961, 0, 0,  0]

#Splitting the Training and Testing data
from sklearn.model_selection import train_test_split

X = dataframe[features]
y = dataframe[predict]

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=test_size,
                                                    random_state=random_state)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Accuracy Score % : ",score*100)

model.predict([PredictFeatures])

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Fraud', 'Non Fraud'],
    mode='classification'
)

'''
exp = explainer.explain_instance(
    data_row=X_test.iloc[1],
    predict_fn=model.predict_proba
)
'''

Xp = pd.DataFrame(np.array([PredictFeatures]),columns=features)

exp = explainer.explain_instance(
    data_row=Xp.iloc[0],
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True)