# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:50:02 2020

@author: masud
"""

import glob
import time
import pandas as pd
import numpy as np
# from xml.dom import minidom
from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

import pandas as pd
df = pd.read_csv("QueryResults.csv")
df.head()

print(df.head())

## Common and Uncommon problem class ratio 
df = df['Class'].value_counts()
print('Class 0:', df[0])
print('Class 1:', df[1])
print('Proportion:', round(df[0] / df[1], 2), ': 1')

## Visual view of common and uncommon class 
df.plot(kind='bar', title='Count (target)');

## Remove lower important feature



