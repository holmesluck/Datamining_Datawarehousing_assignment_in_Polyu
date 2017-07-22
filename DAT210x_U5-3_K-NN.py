#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121 Lab on 2017 JUN 24

@author: King
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score


# import the data set
data = pd.read_csv('./test1.csv',index_col=0)
data_test = pd.read_csv('./test1.1.csv',index_col=0)

# all the feature = ['Customer No.', 'Sex	Average Transactions','Average Monthly Payment','Average months in Silver',	'Decision']
feature_test =  ['Sex','Average Transactions','Average Monthly Payment','Average months in Silver']
feature_label = ['Decision']

# normalization of the data preprocessing.
def MaxMinNormalizaiton(x):
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    return x


data[feature_test] = MaxMinNormalizaiton(data[feature_test])
data_test[feature_test] = MaxMinNormalizaiton(data_test[feature_test])
print (data)
print(data_test[feature_test])
# data_test[feature_test]=MaxMinNormalizaiton(data_test[feature_test])
#create the knn obeject and fit the data set
model = KNeighborsClassifier(n_neighbors=5)
model.fit(data[feature_test], data[feature_label])
#get the prediction value
predictions = model.predict(data_test[feature_test])
print(predictions)

# print(model.predict_proba(data_test))
#
# print(model.score(data_test, label_test))
#
# print(accuracy_score(label_test, predictions))
# print(accuracy_score(label_test, predictions, normalize=False))
#
# print(metrics.confusion_matrix(predictions, label_test))
# print(metrics.classification_report(label_test, predictions))