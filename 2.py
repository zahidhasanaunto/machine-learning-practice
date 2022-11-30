# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("./data/diabetes.csv", header=None, names=col_names)
test_pima = pd.read_csv("./data/diabetes_test.csv", header=None, names=col_names)

for column in pima.columns:
    temp_new = le.fit_transform(pima[column].astype('category'))
    pima.drop(labels=[column], axis="columns", inplace=True)
    pima[column] = temp_new

for column in test_pima.columns:
    temp_new = le.fit_transform(test_pima[column].astype('category'))
    test_pima.drop(labels=[column], axis="columns", inplace=True)
    test_pima[column] = temp_new

pima.head()
test_pima.head()

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# print(X)
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

print (X_test)
print (test_pima)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Prediction:", y_pred)
