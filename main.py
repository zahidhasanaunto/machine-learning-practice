import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def importData(path):
    balance_data = pd.read_csv(path, sep= ',', header = None)
    return balance_data

def splitDataset(balance_data):
  
    X = balance_data.values[:, 1:6]
    Y = balance_data.values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
      
    return X, Y, X_train, X_test, y_train, y_test
      
def trainUsingGini(X_train, X_test, y_train):
  
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)

    return clf_gini
      
def trainUsingEntropy(X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(X_train, y_train)

    return clf_entropy
  
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred
      
def findAccuracy(y_test, y_pred):
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
  
def main():

    data = importData('./data/12000.csv')

    X, Y, X_train, X_test, y_train, y_test = splitDataset(data)

    clfGini = trainUsingGini(X_train, X_test, y_train)
    clfEntropy = trainUsingEntropy(X_train, X_test, y_train)

    # "HTML", "CSS", "JavaScript", "NodeJS", "MongoDB"
    NEW_X_test = np.array(
        [
            [2,2,2,5,4], # BE
            [4,5,4,3,2], # FE
            [2,2,2,4,5], # DB,
            [8, 7, 9, 5, 4] # Mid FE
        ]
    )
      
    # Prediction using gini
    print("Results Using Gini:")
    yPredGini = prediction(X_test, clfGini)
    findAccuracy(y_test, yPredGini)

    yPredGiniNew = prediction(NEW_X_test, clfGini)
    print("Predicted From New Case: ", yPredGiniNew)

    print('=====================================')
      
    # # Prediction using entropy
    print("Results Using Entropy:")
    yPredEntropy = prediction(X_test, clfEntropy)
    findAccuracy(y_test, yPredEntropy)

    yPredEntropyNew = prediction(NEW_X_test, clfEntropy)
    print("Predicted From New Case: ", yPredEntropyNew)
      
      
# Calling main function,
if __name__=="__main__":
    main()