#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121 Lab on 2017 JUN 24

@author: King
"""

import numpy as np
import pandas as pd
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn import tree
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from IPython.display import Image
from sklearn.tree import export_graphviz


# data   = [[0],[1],[2],[3],[4], [5],[6],[7],[8],[9]]  # input dataframe samples
# labels = [0,0,0,0,0, 1,1,1,1,1]  # the function we're training is " >4 "



# data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.5, random_state=7)

data = pd.read_csv('./decisiontree.csv')

data_test = pd.read_csv('./decisiontree_test.csv')


# ohv_data = pd.get_dummies(data,sparse=True)
# ohv_data_test = pd.get_dummies(data_test,sparse=True)

#
# print(ohv_data)
# print(ohv_data_test)

model = tree.DecisionTreeClassifier(max_depth=5, criterion="entropy")

train_feature = ['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism']
label_feature = ['Recommendation']
test_feature = ['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism']
test_label_feature = ['Recommendation']

Tear_mapping = {'Reduced':0,'Normal':1}
Sex_mapping ={'M':0,'F':1}
Age_mapping = {'Young':0,'Middle':1,'Old':2}
SP_mapping = {'Myope':0,'Hypermetrope':1}
As_mapping = {'Yes':1,'No':0}
R_mapping = {'Lifestyle':0,'Street':1,'Polarized':2}
data['Tear Production Rate'] = data['Tear Production Rate'].map(Tear_mapping)
data['Sex'] = data['Sex'].map(Sex_mapping)
data['Age'] = data['Age'].map(Age_mapping)
data['Spectacle Prescription'] = data['Spectacle Prescription'].map(SP_mapping)
data['Astigmatism'] = data['Astigmatism'].map(As_mapping)
data['Recommendation'] = data['Recommendation'].map(R_mapping)
data_test['Tear Production Rate'] = data_test['Tear Production Rate'].map(Tear_mapping)
data_test['Sex'] = data_test['Sex'].map(Sex_mapping)
data_test['Age'] = data_test['Age'].map(Age_mapping)
data_test['Spectacle Prescription'] = data_test['Spectacle Prescription'].map(SP_mapping)
data_test['Astigmatism'] = data_test['Astigmatism'].map(As_mapping)
data_test['Recommendation'] = data_test['Recommendation'].map(R_mapping)
data_train = data[train_feature]
label_train = data[label_feature]
data_test_set = data_test[test_feature]
label_test_set = data_test[test_label_feature]


# print(data_train)
# print(label_train)
# print(data_test_set)
# print(label_test_set)

model.fit(data_train,label_train)


predictions = model.predict(data_test_set)
print(predictions)

print(model.score(data_test_set, label_test_set))

# print(accuracy_score(label_test_set, predictions))
# print(accuracy_score(label_test_set, predictions, normalize=False))
#
# print(metrics.confusion_matrix(predictions, label_test_set))

dotfile = open("./dtree.dot", 'w')
# dot_data = tree.export_graphviz(model.tree_, out_file = None, feature_names = ['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism']
#
#                                  )
dot_data = tree.export_graphviz(model,out_file=dotfile,feature_names=['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism','Recommandation'])
                                # class_names=label_feature,filled=True,rounded=True
                                # ,special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
dotfile.close()

"""
Open the .dot file created with a text editor.
Copy all the text in the .dot file and paste to the textbox at http://webgraphviz.com/ (Web Service of Graphviz)
"""



# def vistualize_tree (tree,feature_names):
#  with open("dt.dot",'w') as f:
#       export_graphviz(tree, out_file=f, feature_names=feature_names)
#
#       command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
#       try:
#           pydotplus.subprocess.check_call(command)
#       except:
#           exit("NO way")
#
#
# vistualize_tree(model,train_feature)