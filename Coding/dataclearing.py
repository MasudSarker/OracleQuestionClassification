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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer

#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

tokenizer = ToktokTokenizer()
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

import pandas as pd
df = pd.read_csv("QueryResults.csv")
df.head()


## Ratio of missing values per columns
plt.figure(figsize=(5, 5))
df.isnull().mean(axis=0).plot.barh()
plt.title("Ratio of missing values per columns")

## Remove lower important feature

def remove_html(text):
    # Remove html and convert to lowercase
    return re.sub(r"\<[^\>]\>", "", text).lower()

def remove_stopwords(text):    
    # tokenize the text
    words = tokenizer.tokenize(text)
    
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def remove_punc(text):
    #tokenize
    tokens = tokenizer.tokenize(text)
    
    # remove punctuations from each token
    tokens = list(map(lambda token: re.sub(r"[^A-Za-z0-9]+", " ", token).strip(), tokens))
    
    # remove empty strings from tokens
    tokens = list(filter(lambda token: token, tokens))
    
    return ' '.join(map(str, tokens))

def stem_text(text):
    #tokenize
    tokens = tokenizer.tokenize(text)
    
    # stem each token
    tokens = list(map(lambda token: stemmer.stem(token), tokens))
    
    return ' '.join(map(str, tokens))


# apply preprocessing to title and body
## Remove puntuation and preposition from text(title and body)
df['Title'] = df['Title'].apply(lambda x: remove_html(x))
df['Title'] = df['Title'].apply(lambda x: remove_stopwords(x))
df['Title'] = df['Title'].apply(lambda x: remove_punc(x))
df['Title'] = df['Title'].apply(lambda x: stem_text(x))

# apply preprocessing to title and body
df['Body'] = df['Body'].apply(lambda x: remove_html(x))
df['Body'] = df['Body'].apply(lambda x: remove_stopwords(x))
df['Body'] = df['Body'].apply(lambda x: remove_punc(x))
df['Body'] = df['Body'].apply(lambda x: stem_text(x))


## Text feature vectorizer

vectorizer_title = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                       max_features=4)


vectorizer_body = TfidfVectorizer(
                                    analyzer = 'word', 
                                    strip_accents = None, 
                                    encoding = 'utf-8', 
                                    preprocessor=None, 
                                    max_features=10)

#df['Title'] = vectorizer_title.fit_transform(df['Title'])
#df['Body'] = vectorizer_title.fit_transform(df['Body'].values.astype('U')).toarray()

#print(df['Title'])
#datav= df.head()

vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
 
texttile = vectorizer.fit_transform(df['Title'].values.astype('U')).toarray()
textbody = vectorizer.fit_transform(df['Body'].values.astype('U')).toarray()
# print(posts)

texttile=pd.DataFrame(texttile)

textbody=pd.DataFrame(textbody)


df=pd.concat([df,texttile,textbody],axis=1)

## Remove less important features
df = df.drop(['Body', 'Title','Tags','creationDate','LastActivityDate','CurrentDate','AcceptedAnswerId'],axis = 1,inplace=False)





