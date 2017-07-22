import pandas as pd

#Q1_a_b
# load all the data by using the pandas api
data_set = pd.read_csv('Apriori.csv',index_col=0)
#check the loading result
print(data_set)

#Q1_c
# preprocessing the data_set
T_mapping = {"T":1}
data_set['Biscuit A'] = data_set['Biscuit A'].map(T_mapping)
data_set['Biscuit A'] = data_set['Biscuit A'].fillna(0)
data_set['Nut B'] = data_set['Nut B'].map(T_mapping)
data_set['Nut B'] = data_set['Nut B'].fillna(0)
data_set['Noodle C'] = data_set['Noodle C'].map(T_mapping)
data_set['Noodle C'] = data_set['Noodle C'].fillna(0)
data_set['Shampoo D'] = data_set['Shampoo D'].map(T_mapping)
data_set['Shampoo D'] = data_set['Shampoo D'].fillna(0)
data_set['Bread E'] = data_set['Bread E'].map(T_mapping)
data_set['Bread E'] = data_set['Bread E'].fillna(0)
data_set['Diaper F'] = data_set['Diaper F'].map(T_mapping)
data_set['Diaper F'] = data_set['Diaper F'].fillna(0)
data_set['Beer G'] = data_set['Beer G'].map(T_mapping)
data_set['Beer G'] = data_set['Beer G'].fillna(0)
data_set['Milk H'] = data_set['Milk H'].map(T_mapping)
data_set['Milk H'] = data_set['Milk H'].fillna(0)
#check the data_set preformance after the preprocessing
print(data_set)