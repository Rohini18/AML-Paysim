import pandas as pd

#readFilePath = "TransformedData.csv"
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


from sklearn.ensemble import RandomForestClassifier
print("\nTraining using RandomForestClassifier model")
Model = RandomForestClassifier(n_estimators=100,
                               n_jobs=2,
                               random_state=random_state)
Model.fit(X_train, y_train)
print("Accuracy of Training set data : ",round(Model.score(X_train,y_train)*100,3),"%")
print("Accuracy of Testing set data  : ",round(Model.score(X_test,y_test)*100,3),"%")


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics

predicted = Model.predict(X_train)
actual = y_train

precision = precision_score(actual, predicted)
recall = recall_score(actual, predicted)
f1 = f1_score(actual, predicted)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Non Fruad", "Fruad"])
                                   
import matplotlib.pyplot as plt
cm_display.plot()
plt.show()




