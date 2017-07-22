#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121

Hierarchical Agglomerative Single-Linkage
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# import the data from the data set excel table
data = pd.read_csv('./k-means.csv',index_col=0)

# feature name declaration
feature_to_scale =['Age ','Monthly Income ','Service Plan ','Extra Usage ']

training_feature = ['Age ','Sex ','Monthly Income ','Marital Status ','Service Plan ','Extra Usage ']

# convert the string value into int in the data set
Sex_mapping = {'FEMALE ':0,'MALE ':1}
MS_mapping = {'YES ':1, 'NO ':0}
data['Sex '] = data['Sex '].map(Sex_mapping)
data['Marital Status '] = data['Marital Status '].map(MS_mapping)



# normalization of the data preprocessing.
def MaxMinNormalizaiton(x):
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    return x

# do the normalization
data[feature_to_scale] = MaxMinNormalizaiton(data[feature_to_scale])

data_train = data[training_feature]



#input the data into the AgglomerativeClustering algorithm
model = AgglomerativeClustering(n_clusters=2)

model.fit(data_train)


# get the label and centroids
labels = model.fit_predict(data_train)

children = model.children_

leaves = model.n_leaves_

component = model.n_components_
# get the index of the each group
group1_index = np.argwhere(labels==0)+1
group2_index = np.argwhere(labels==1)+1
# print out
print 'group no.1 contains:', group1_index
print 'group no.2 contains:', group2_index
print(labels)
print(children)
print(leaves)
print(component)

