#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121 Lab on 2017 JUN 24

@author: King
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
test_data = pd.read_csv("./test.csv", index_col=0)

# student_dataset = pd.read_csv("./students.data", index_col=0)
# print(student_dataset)
#
# data_train = student_dataset[['G3', 'G2', 'G1']]

# real = Normalizer()

feature_to_scale =['Age','Monthly Income','Service Plan','Extra Usage']

training_feature = ['Age','Sex','Monthly Income','Marital Status','Service Plan','Extra Usage']
# real_data = real.fit_transform(test_data[feature_to_scale])


# normalization of the data preprocessing.
def MaxMinNormalizaiton(x):
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    return x

temp = MaxMinNormalizaiton(test_data[feature_to_scale])

test_data[feature_to_scale] = temp

# print (test_data['Sex'])

data_train = test_data[training_feature]

# print (data_train)
initial_center = data_train[0:2]
print(initial_center)
model = KMeans(n_clusters=2,init='k-means++')

model.fit(data_train)

labels = model.predict(data_train)

centroids = model.cluster_centers_

print(centroids)
print(labels)