#!usr/bin/env python3

import os
import numpy as np
import pandas as pd
import glob
import pickle as c
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from string import punctuation

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# tokenize

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):

    non_words = list(punctuation)
    non_words.extend(['¿', '¡'])
    stemmer = SnowballStemmer('spanish', ignore_stopwords=True)

    tweets = []
    for t in text:
        palabras = t.split(' ')
        text = ''.join([c for c in palabras if c not in non_words])
        tokens =  word_tokenize(text)
        # stem
        try:
            stems = stem_tokens(tokens, stemmer)
        except Exception as e:
            print(e)
            print(text)
            stems = ''
        str = ""
        tweets.append(str.join(stems))
    print(len(tweets))
    return tweets

#Remove stop words from tweets
def remove_stop_words(df_tweets):
    tweets = []
    for t in df_tweets:
        palabras = t.split(' ')
        filtradas = [word for word in palabras if word not in stopwords.words('spanish')]
        str = ""
        tweets.append(str.join(filtradas))
    return tweets


#read in the dataset
path = r'data_set'
files = glob.glob(path + "/2018-EI-reg-Es*.txt")
li = []

for filename in files:
    df = pd.read_csv(filename,  sep = "	", header=None, skiprows=1)
    li.append(df)

df = pd.concat(li)

df.columns = ['id', 'Tweet', 'AffectDimension', 'IntensityScore']
df = df.drop(columns=['id', 'IntensityScore'])

df['Tweet'] = remove_stop_words(df['Tweet'])

df['Tweet'] = tokenize(df['Tweet'])

#label encode the values before passing the features in the dataset.
le = LabelEncoder()
for i in range(2):
    df.iloc[:,i] = le.fit_transform(df.iloc[:,i])

X = df['Tweet']
y = df['AffectDimension']


X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, stratify=y)

#create new a knn model
knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(15, 25), 'algorithm': ['auto'], 'leaf_size': np.arange(1, 18), 'p': [1,2], 'n_jobs': [-1], 'weights': ['uniform', 'distance']}

#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)

# convert x into 2D matrix
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_
y_true, y_pred = y_test, knn_gs.predict(X_test)
print("-------------K-Neighbords-------------")
print(knn_gs.best_params_)
print(knn_gs.best_score_)
print(classification_report(y_true, y_pred))


#create a new random forest classifier
rf = RandomForestClassifier()
#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv=5)
rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_
y_true, y_pred = y_test, rf_gs.predict(X_test)
print("-------------Random Forest-------------")
print(rf_gs.best_params_)
print(rf_gs.best_score_)
print(classification_report(y_true, y_pred))


#create a new logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_true, y_pred = y_test, log_reg.predict(X_test)
print("-------------Logistic Regresion-------------")
print(classification_report(y_true, y_pred))

print("-------------Accuracy scores-------------")
print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))


#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')

ensemble.fit(X_train, y_train)
print("--------------Ensemble------------")
print('ensemble: {}'.format(ensemble.score(X_test, y_test)))
