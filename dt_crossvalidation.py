from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold, StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from numpy import mean
from numpy import std
from scipy.stats import sem


def evaluate_model_stratified_kfold(model, x, y, split_number, with_shuffle=False):
    cv = StratifiedKFold(n_splits=split_number, random_state=None, shuffle=with_shuffle)
    #for train_index, test_index in cv.split(x, y):
     #   print("TRAIN:", train_index, "TEST:", test_index)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def evaluate_model_repeated_kfold(model, x, y, repeats):
    cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def evaluate_model_kfold(model, x, y, split_number, with_shuffle):
    cv = KFold(n_splits=split_number, random_state=1, shuffle=with_shuffle)
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


breast_cancer = load_breast_cancer()
x = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
decision_tree_model = DecisionTreeClassifier()

scores = evaluate_model_kfold(decision_tree_model, x, y, 10, True)
print('KFold : mean accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

scores = evaluate_model_repeated_kfold(decision_tree_model, x, y, 5)
print('RepeatedKFold : with %d repeats - mean accuracy=%.4f se=%.3f' % (5, mean(scores), sem(scores)))

scores = evaluate_model_stratified_kfold(decision_tree_model, x, y, 2)
print('StratifiedKFold : mean accuracy=%.4f se=%.3f' % (mean(scores), sem(scores)))




