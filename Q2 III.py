#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created for COMP5121

naive baysian algorithm
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# read and import the data set
data = pd.read_csv('./combine dataset.csv')
feature = ['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism']
classlabel = ['Recommendation']

# convert the string value into int in the data set
Tear_mapping = {'Reduced ':0,'Normal ':1}
Sex_mapping ={'M ':0,'F ':1}
Age_mapping = {'Young ':0,'Middle ':1,'Old ':2}
SP_mapping = {'Myope ':0,'Hypermetrope ':1}
As_mapping = {'Yes ':1,'No ':0}
R_mapping = {'Lifestyle ':0,'Street ':1,'Polarized ':2}
data['Tear Production Rate'] = data['Tear Production Rate'].map(Tear_mapping)
data['Sex'] = data['Sex'].map(Sex_mapping)
data['Age'] = data['Age'].map(Age_mapping)
data['Spectacle Prescription'] = data['Spectacle Prescription'].map(SP_mapping)
data['Astigmatism'] = data['Astigmatism'].map(As_mapping)
data['Recommendation'] = data['Recommendation'].map(R_mapping)

train_data = data[feature]
train_data_label = data[classlabel]


# seperate the data set to train and test set
train_data = data[0:15]

test_data = data[15:]

# seperate training data set to two array,one is data set, the other is class label
train_data_set = train_data[feature]
train_data_label = train_data[classlabel]


# seperate testing data set to two array,one is data set, the other is class label
test_data_set = test_data[feature]
test_data_label = test_data[classlabel]



# perform the classification using the Naive Baysian approach
model = GaussianNB()
model.fit(train_data_set, train_data_label)

predictions = model.predict(test_data_set)
print(predictions)
print(model.score(test_data_set, test_data_label))