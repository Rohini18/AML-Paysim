
import pandas as pd

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
#alert
#dataframe = dataframe[:10000]

print("Count of data           : ",len(dataframe))
print("Count of fraud data     : ",dataframe['isFraud'].value_counts()[1])
print("Count of non fraud data : ",dataframe['isFraud'].value_counts()[0])



from sklearn.model_selection import train_test_split

X = dataframe[features]
y = dataframe[predict]



X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=test_size,
                                                    random_state=random_state)
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
'''


from sklearn.linear_model import LinearRegression
print("\nTraining using Linear Regression model")
Model = LinearRegression()
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")

from sklearn.linear_model import LogisticRegression
print("\nTraining using Logistic Regression model")
Model = LogisticRegression()
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")

'''
from sklearn.linear_model import LogisticRegressionCV
print("\nTraining using Logistic Regression CV model")
Model = LogisticRegressionCV()
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")


from sklearn.neighbors import KNeighborsClassifier
print("\nTraining using KNeighborsClassifier model")
Model = KNeighborsClassifier(n_neighbors=5)
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")

from sklearn.tree import DecisionTreeClassifier
print("\nTraining using DecisionTreeClassifier model")
Model = DecisionTreeClassifier(criterion='gini',
                               max_depth=None,
                               random_state=random_state)
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")


from sklearn.ensemble import RandomForestClassifier
print("\nTraining using RandomForestClassifier model")
Model = RandomForestClassifier(n_estimators=100,
                               n_jobs=2,
                               random_state=random_state)
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")



from sklearn import svm
print("\nTraining using SVM model")
#Model = svm.SVC(gamma='scale', kernel='rbf')
#Model = svm.SVC(gamma='scale', kernel='poly')
Model = svm.SVC(gamma='scale', kernel='linear')
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")



from sklearn.ensemble import AdaBoostClassifier
print("\nTraining using AdaBoostClassifier model")
base_estimator = DecisionTreeClassifier(max_depth=3)
Model = AdaBoostClassifier(base_estimator=base_estimator,
                           n_estimators=100,
                           random_state=1)
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")


from sklearn.ensemble import GradientBoostingClassifier
print("\nTraining using GradientBoostingClassifier model")
Model = GradientBoostingClassifier(n_estimators=1000,
                                   criterion='mse',
                                   loss='exponential',
                                   max_depth=3,
                                   learning_rate=1.0,
                                   random_state=1)
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")


from xgboost import XGBClassifier
print("\nTraining using XGBClassifier model")
Model = XGBClassifier(n_estimators=1000, learning_rate=1)
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")
'''


