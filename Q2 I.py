# Q2 b)
import pandas as pd

data = pd.read_csv('./combine dataset.csv')
train_data = data[0:15]
print(train_data)
test_data = data[15:]
print (test_data)
feature = ['Tear Production Rate','Sex','Age','Spectacle Prescription','Astigmatism']
classlabel = ['Recommendation']
# seperate training data set to two array,one is data set, the other is class label
train_data_set = train_data[feature]
train_data_label = train_data[classlabel]
# seperate testing data set to two array,one is data set, the other is class label
test_data_set = test_data[feature]
test_data_label = test_data[classlabel]
print (train_data_set)
print (train_data_label)
print (test_data_label)
print (test_data_set)