#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121

K-means
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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


# specify the initial centroids
init_centroids_1 = np.array(data[0:2])
a = np.array(data[0:1])
b = np.array(data[-1:])

init_centroids_2 = np.vstack((a,b))

#input the data into the kmeans algorithm
def inputdata(x,a):
 model = KMeans(n_clusters=2,init=x,n_init=a)

 model.fit(data_train)


# get the label and centroids
 labels = model.predict(data_train)

 centroids = model.cluster_centers_
# get the index of the each group
 group1_index = np.argwhere(labels==0)+1
 group2_index = np.argwhere(labels==1)+1
# show the group and centeroids
 print 'centroids is:',centroids
 print 'group no.1 contains:',group1_index
 print 'group no.2 contains:',group2_index

# calculate the result of different initial centroids
print('-----when the initial centroids is 1st and 2nd record of the table-----')
inputdata(init_centroids_1,1)
print('-----when the initial centroids is 1st and 2nd record of the table-----')
inputdata(init_centroids_2,1)
print('-----when use the k-means++ default centroids-----')
inputdata('k-means++',10)
print('-----when use the default centroids-----')
model = KMeans(n_clusters=2)

model.fit(data_train)


# get the label and centroids
labels = model.predict(data_train)

centroids = model.cluster_centers_
# get the index of the each group
group1_index = np.argwhere(labels==0)+1
group2_index = np.argwhere(labels==1)+1
# show the group and centeroids
print 'centroids is:',centroids
print 'group no.1 contains:',group1_index
print 'group no.2 contains:',group2_index