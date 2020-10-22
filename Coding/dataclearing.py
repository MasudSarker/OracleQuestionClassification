# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:17:35 2020

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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

import pandas as pd
df = pd.read_csv("QueryResults.csv")
df.head()


## Ratio of missing values per columns
plt.figure(figsize=(5, 5))
df.isnull().mean(axis=0).plot.barh()
plt.title("Ratio of missing values per columns")

## Remove lower important feature



