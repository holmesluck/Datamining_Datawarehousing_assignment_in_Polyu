#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121 Lab on 2017 JUN 24

@author: King
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score

#X_train = pd.DataFrame([ [0], [1], [2], [3] ])
#y_train = [0, 0, 1, 1]
data   = [[0],[1],[2],[3],[4], [5],[6],[7],[8],[9]]  # input dataframe samples
labels = [0,0,0,0,0, 1,1,1,1,1]  # the function we're training is " >4 "

data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=5)


print (data_train)
# model = GaussianNB()
# model.fit(data_train, label_train)
#
# predictions = model.predict(data_test)
# print(model.predict(data_test))
# print(model.predict_proba(data_test))

# print(model.score(data_test, label_test))

# print(accuracy_score(label_test, predictions))
# print(accuracy_score(label_test, predictions, normalize=False))

# print(metrics.confusion_matrix(predictions, label_test))
# print(metrics.classification_report(label_test, predictions))