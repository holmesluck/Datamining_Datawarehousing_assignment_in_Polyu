#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121

k_NN algorithm
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# import the data set
data = pd.read_csv('./Knn.csv',index_col=0)
data_test = pd.read_csv('./Knntest2.csv',index_col=0)

# feature in the data
feature = ['Sex','Average Monthly Payment',
           'Average No. of months in Silver']
label = ['Average No. of Transactions']

# convert the string value into int in the data set
Sex_mapping = {'F ':0 , 'M ':1}
data['Sex'] =  data['Sex'].map(Sex_mapping)
data_test['Sex'] =  data_test['Sex'].map(Sex_mapping)



# normalization of the data preprocessing.
def MaxMinNormalizaiton(x):
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    return x

# excute the normalization
# add the test data into the train data set to do the normalization
new = data.append(data_test)
# train data set normalization
data[feature] = MaxMinNormalizaiton(data[feature])
# data[label] = MaxMinNormalizaiton(data[label])
# test data set normalizaion
temp = MaxMinNormalizaiton(new[feature])
data_test[feature] = temp[-1:]

#create the knn obeject and fit the data set
model = KNeighborsClassifier(n_neighbors=5)
model.fit(data[feature], data[label])
#get the prediction value
predictions = model.predict(data_test[feature])
print 'prediction value is ',predictions
# print(model.predict_proba(data_test[feature]))


