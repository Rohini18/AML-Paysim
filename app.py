from flask import Flask, render_template, redirect, url_for, request, Blueprint, jsonify, flash

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


import lime
from lime import lime_tabular

app = Flask(__name__)

readFilePath = "TransformedData.csv"
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
dataframe = pd.read_csv(readFilePath)

print("Count of data           : ",len(dataframe))
print("Count of fraud data     : ",dataframe['isFraud'].value_counts()[1])
print("Count of non fraud data : ",dataframe['isFraud'].value_counts()[0])


#alert
dataframe = dataframe

X = dataframe[features]
y = dataframe[predict]



X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=test_size,
                                                    random_state=random_state)




print("\nTraining using model")
Model1 = LogisticRegressionCV()
Model2 = RandomForestClassifier(n_estimators=100,
                                n_jobs=2,
                                random_state=random_state)




Model1.fit(X_train.values, y_train)
print("Accuracy of Training set data : ",round(Model1.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model1.score(X_test,y_test)*100,3),"%")

Model2.fit(X_train.values, y_train)
print("Accuracy of Training set data : ",round(Model2.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model2.score(X_test,y_test)*100,3),"%")

Model = Model1

@app.route('/', methods=["POST", "GET"])
def test():
    global dataframe
    global Model
    fraud = ""
    predict = ""
    if request.method == 'POST':
        predict = True
        column = 'type'
        trans_type = list(dataframe.query(column+'=="'+request.form['trans_type']+'"')[-1:]["trans_type"])[0]
        amount = request.form['amount']
        column = 'nameOrig'
        trans_nameOrig = list(dataframe.query(column+'=="'+request.form['trans_nameOrig']+'"')[-1:]["trans_nameOrig"])[0]
        oldbalanceOrg = request.form['oldbalanceOrg']
        newbalanceOrig = request.form['newbalanceOrig']
        column = 'nameDest'
        trans_nameDest = list(dataframe.query(column+'=="'+request.form['trans_nameDest']+'"')[-1:]["trans_nameDest"])[0]
        oldbalanceDest = request.form['oldbalanceDest']
        newbalanceDest = request.form['newbalanceDest']
        isFlaggedFraud = request.form['isFlaggedFraud']
        pfeatures = [int(trans_type),
                    int(amount),
                    int(trans_nameOrig),
                    int(oldbalanceOrg),
                    int(newbalanceOrig),
                    int(trans_nameDest),
                    int(oldbalanceDest),
                    int(newbalanceDest),
                    int(isFlaggedFraud)]
        print("features  : ",pfeatures)
        print("Model  : ",Model.predict([pfeatures]))
        fraud = Model.predict([pfeatures])
        if fraud == 0:
            fruad = False
        else:
            fruad = True
    return render_template("Base.html",
                           fraud = fraud,
                           predict = predict)



@app.route('/setmodel1')
def setmodel1():
    global Model1
    global Model
    Model = Model1
    return " Model set to model 1"


@app.route('/setmodel2')
def setmodel2():
    global Model2
    global Model
    Model = Model2
    return " Model set to model 2"


@app.route('/api/predict', methods=["POST", "GET"])
def api_predict():
    global dataframe
    global Model
    fraud = ""
    predict = ""
    if request.method == 'POST':
        predict = True
        column = 'type'
        trans_type = list(dataframe.query(column+'=="'+request.form['trans_type']+'"')[-1:]["trans_type"])[0]
        amount = request.form['amount']
        column = 'nameOrig'
        trans_nameOrig = list(dataframe.query(column+'=="'+request.form['trans_nameOrig']+'"')[-1:]["trans_nameOrig"])[0]
        oldbalanceOrg = request.form['oldbalanceOrg']
        newbalanceOrig = request.form['newbalanceOrig']
        column = 'nameDest'
        trans_nameDest = list(dataframe.query(column+'=="'+request.form['trans_nameDest']+'"')[-1:]["trans_nameDest"])[0]
        oldbalanceDest = request.form['oldbalanceDest']
        newbalanceDest = request.form['newbalanceDest']
        isFlaggedFraud = request.form['isFlaggedFraud']
        pfeatures = [int(trans_type),
                    int(amount),
                    int(trans_nameOrig),
                    int(oldbalanceOrg),
                    int(newbalanceOrig),
                    int(trans_nameDest),
                    int(oldbalanceDest),
                    int(newbalanceDest),
                    int(isFlaggedFraud)]
        print("features  : ",pfeatures)
        print("Model  : ",Model.predict([pfeatures]))
        fraud = Model.predict([pfeatures])
        if fraud == 0:
            resp = '{ "result" : "Not a fraud Transaction" }'
        else:
            resp = '{ "result" : "Alert! fraud Transaction" }'
    return resp

@app.route('/explain', methods=["POST", "GET"])
def explain():
    global dataframe
    global Model
    global X_train
    global features
    if request.method == 'GET':
        return render_template("Explain.html")
    if request.method == 'POST':
        predict = True
        column = 'type'
        trans_type = list(dataframe.query(column+'=="'+request.form['trans_type']+'"')[-1:]["trans_type"])[0]
        amount = request.form['amount']
        column = 'nameOrig'
        trans_nameOrig = list(dataframe.query(column+'=="'+request.form['trans_nameOrig']+'"')[-1:]["trans_nameOrig"])[0]
        oldbalanceOrg = request.form['oldbalanceOrg']
        newbalanceOrig = request.form['newbalanceOrig']
        column = 'nameDest'
        trans_nameDest = list(dataframe.query(column+'=="'+request.form['trans_nameDest']+'"')[-1:]["trans_nameDest"])[0]
        oldbalanceDest = request.form['oldbalanceDest']
        newbalanceDest = request.form['newbalanceDest']
        isFlaggedFraud = request.form['isFlaggedFraud']
        featuresv = [int(trans_type),
                    int(amount),
                    int(trans_nameOrig),
                    int(oldbalanceOrg),
                    int(newbalanceOrig),
                    int(trans_nameDest),
                    int(oldbalanceDest),
                    int(newbalanceDest),
                    int(isFlaggedFraud)]
        features = ["trans_type",
                    "amount",
                    "trans_nameOrig",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "trans_nameDest",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "isFlaggedFraud"]
        PredictFeatures = [4,181,3101,181,0,8961,0,0,0]
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=features,
            mode='classification')
        Xp = pd.DataFrame(np.array([PredictFeatures]),columns=features)
        exp = explainer.explain_instance(data_row=Xp.iloc[0].values, predict_fn=Model.predict)
        return exp.as_html()
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=8080,
            debug = True,
            threaded=True)


'''
@app.route('/explain', methods=["POST", "GET"])
def explain():
    global dataframe
    global Model
    global X_train
    global features
    if request.method == 'GET':
        return render_template("Explain.html")
    if request.method == 'POST':
        predict = True
        column = 'type'
        trans_type = list(dataframe.query(column+'=="'+request.form['trans_type']+'"')[-1:]["trans_type"])[0]
        amount = request.form['amount']
        column = 'nameOrig'
        trans_nameOrig = list(dataframe.query(column+'=="'+request.form['trans_nameOrig']+'"')[-1:]["trans_nameOrig"])[0]
        oldbalanceOrg = request.form['oldbalanceOrg']
        newbalanceOrig = request.form['newbalanceOrig']
        column = 'nameDest'
        trans_nameDest = list(dataframe.query(column+'=="'+request.form['trans_nameDest']+'"')[-1:]["trans_nameDest"])[0]
        oldbalanceDest = request.form['oldbalanceDest']
        newbalanceDest = request.form['newbalanceDest']
        isFlaggedFraud = request.form['isFlaggedFraud']
        featuresv = [int(trans_type),
                    int(amount),
                    int(trans_nameOrig),
                    int(oldbalanceOrg),
                    int(newbalanceOrig),
                    int(trans_nameDest),
                    int(oldbalanceDest),
                    int(newbalanceDest),
                    int(isFlaggedFraud)]
        analyseDF = pd.DataFrame(np.array([featuresv]),columns=features)
        print("analyseDF : ", analyseDF)
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=features,
            mode='classification')
        exp = explainer.explain_instance(
            data_row=analyseDF.iloc[0].values,
            predict_fn=Model.predict)
        return exp.as_html()
'''
