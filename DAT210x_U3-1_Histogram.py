#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for COMP5121 Lab on 2017 JUN 24

@author: King
"""
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot') # Look Pretty
# If the above line throws an error, use plt.style.use('ggplot') instead

student_dataset = pd.read_csv("./students.data", index_col=0)

my_series = student_dataset.G3
my_dataframe = student_dataset[['G3', 'G2', 'G1']] 

my_series.plot.hist(alpha=0.5)
my_dataframe.plot.hist(alpha=0.5)

plt.show()
