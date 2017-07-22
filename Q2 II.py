#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created for COMP5121

decision tree
"""
import pandas as pd
from sklearn import tree




# read and import the data set
data = pd.read_csv('./combine dataset.csv')
feature = ['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism']
classlabel = ['Recommendation']
classname =['Lifestyle','Street','Polarized']
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

feature = ['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism']
classlabel = ['Recommendation']
# seperate training data set to two array,one is data set, the other is class label
train_data_set = train_data[feature]
train_data_label = train_data[classlabel]
# seperate testing data set to two array,one is data set, the other is class label
test_data_set = test_data[feature]
test_data_label = test_data[classlabel]

model = tree.DecisionTreeClassifier(max_depth=5, criterion="entropy")
model.fit(train_data_set,train_data_label)


predictions = model.predict(test_data_set)
print(predictions)

print(model.score(test_data_set, test_data_label))


# create the .dot file to show the tree
dotfile = open("./dtree.dot", 'w')

dot_data = tree.export_graphviz(model,out_file=dotfile,
                                feature_names=feature,
                               )

dotfile.close()

"""
Open the .dot file created with a text editor.
Copy all the text in the .dot file and paste to the textbox at http://webgraphviz.com/ (Web Service of Graphviz)
"""



