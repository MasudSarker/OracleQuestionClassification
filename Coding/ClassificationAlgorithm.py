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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score



import re
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer

#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

tokenizer = ToktokTokenizer()
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

## import rowdataset
import pandas as pd
df = pd.read_csv("RowDataSetTest.csv")
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

## creating a dataframe from list
texttile=pd.DataFrame(texttile)

## creating a dataframe from list
textbody=pd.DataFrame(textbody)

## Concate all feature after preprocessing
df=pd.concat([df,texttile,textbody],axis=1)


## After preprocessing data save into csv file

#f.close()

## End of CSV

## Remove less important features
df = df.drop(['Body', 'Title','Tags','creationDate','LastActivityDate','CurrentDate','AcceptedAnswerId'],axis = 1,inplace=False)

## Scalling column using RobustScaler
df['Id'] = RobustScaler().fit_transform(df['Id'].values.reshape(-1,1))
df['PostTypeId'] = RobustScaler().fit_transform(df['PostTypeId'].values.reshape(-1,1))
#df['creationDate'] = RobustScaler().fit_transform(df['creationDate'].values.reshape(-1,1))
df['Days'] = RobustScaler().fit_transform(df['Days'].values.reshape(-1,1))
df['Score'] = RobustScaler().fit_transform(df['Score'].values.reshape(-1,1))
df['ViewCount'] = RobustScaler().fit_transform(df['ViewCount'].values.reshape(-1,1))
#df['LastActivityDate'] = RobustScaler().fit_transform(df['LastActivityDate'].values.reshape(-1,1))
#df['Tags'] = RobustScaler().fit_transform(df['Tags'].values.reshape(-1,1))
df['AnswerCount'] = RobustScaler().fit_transform(df['AnswerCount'].values.reshape(-1,1))
df['CommentCount'] = RobustScaler().fit_transform(df['CommentCount'].values.reshape(-1,1))



## training and testing dataset devide
# Class count
# Define the prep_data function to extrac features 
def prep_data(df):
    X = df.drop(['Class'],axis=1, inplace=False)  
    X = np.array(X).astype(np.float)
    y = df[['Class']]  
    y = np.array(y).astype(np.float)
    return X,y

# Create X and y from the prep_data function 
X, y = prep_data(df)
y= y.astype(np.int64)
    
#print(Counter(y))

# Apply different ML Classification Algorithm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# ****** LogisticRegression Accuration test
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#clf = svm()
# instantiate classifier with default hyperparameters
clf = SVC()
clf.fit(X_train, y_train)
y_predsvm = clf.predict(X_test)


##LogisticRegression  Machine learning evaluation matrix
accuracy = accuracy_score(y_test, y_pred)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
pre_scor= precision_score(y_test, y_pred)
re_scor = recall_score(y_test, y_pred)
f1_scor = f1_score(y_test, y_pred)

## Random forest classifier algorithm
#Create a Gaussian Classifier
rfclf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
rfclf.fit(X_train, y_train)
y_pred_rf=rfclf.predict(X_test)
## Random Forest Evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision= precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

## End of Random Forest Classifier

## Decision Tree Classifier algorithm
# Create Decision Tree classifer object
dtclf = DecisionTreeClassifier()
# Train Decision Tree Classifer
dtclf = dtclf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred_dt = dtclf.predict(X_test)

## Decision Tree Evaluation
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision= precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)
## End of decision Tree




print("==================== Logistic Regression Evaluation Score ======================")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("\n Precision Score:  %.2f%%" % (pre_scor * 100.0))
print("\n Recall Score:  %.2f%%" % (re_scor * 100.0))
print('\n F1-Measure: %.2f%%' % (f1_scor * 100.0))


## SVM Algorithm Apply
accuracy = accuracy_score(y_test, y_predsvm)
pre_scor= precision_score(y_test, y_predsvm)
re_scor = recall_score(y_test, y_predsvm)
f1_scor = f1_score(y_test, y_predsvm)

print("==================== SVM Evaluation Algorithm ======================")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("\n Precision Score:  %.2f%%" % (pre_scor * 100.0))
print("\n Recall Score:  %.2f%%" % (re_scor * 100.0))
print('\n F1-Measure: %.2f%%' % (f1_scor * 100.0))



print("==================== Random Forest Classifier Algorithm ======================")
print("Accuracy: %.2f%%" % (rf_accuracy * 100.0))
print("\n Precision Score:  %.2f%%" % (rf_precision * 100.0))
print("\n Recall Score:  %.2f%%" % (rf_recall * 100.0))
print('\n F1-Measure: %.2f%%' % (rf_f1 * 100.0))


print("==================== Decision Tree Classifier Algorithm ======================")
print("Accuracy: %.2f%%" % (dt_accuracy * 100.0))
print("\n Precision Score:  %.2f%%" % (dt_precision * 100.0))
print("\n Recall Score:  %.2f%%" % (dt_recall * 100.0))
print('\n F1-Measure: %.2f%%' % (dt_f1 * 100.0))