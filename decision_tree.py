from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

breast_cancer = load_breast_cancer()
x = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
print(x.head())
x.info()
x.describe()  # x.describe(include=['0']) to find string or categorical features

y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
print(y)
print(x[x['worst symmetry'].isna()].head())  # To see data for the feature "worst symmetry" is missing or not
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

fig, _ = plot.subplots(nrows=1, ncols=1, figsize=(100, 50), dpi=300)
tree.plot_tree(decision_tree_model, feature_names=X_train.columns, filled=True)
fig.savefig('tree_model.png')

y_prediction = decision_tree_model.predict(X_test)  # the prediction of the DT model for x_test

# Confusion matrix with comparing actual class for test and predicted class
y_actual = np.array(y_test)
predictions = np.array(y_prediction)
print(confusion_matrix(y_actual, predictions))
tn, fp, fn, tp = confusion_matrix(y_actual, predictions).ravel()
print("True Negative = " + str(tn) + "\nFalse Positive = " + str(fp) + "\nFalse Negative = " + str(fn) +
      "\nTrue Positive = " + str(tp))