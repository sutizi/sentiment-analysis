#!usr/bin/env python3

"""

https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a

"""


import os
import numpy as np
import pandas as pd
import glob
import pickle as c
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder



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

#label encode the values before passing the features in the dataset.
le = LabelEncoder()
for i in range(2):
    df.iloc[:,i] = le.fit_transform(df.iloc[:,i])

#split data into inputs and targets
X = df['Tweet']
y = df['AffectDimension']

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, stratify=y)

#create new a knn model
knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(15, 25), 'algorithm': ['auto'], 'leaf_size': np.arange(1, 18), 'p': [1,2], 'n_jobs': [-1]}

#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)

# convert x into 2D matrix
X_train = X_train.values.reshape(-1,1)

#fit model to training data
knn_gs.fit(X_train, y_train)


#save best model
knn_best = knn_gs.best_estimator_
#check best n_neigbors value
print("Parametros: " + knn_gs.best_params_)
print("Score: " + knn_gs.best_score_)

X_test = X_test.values.reshape(-1,1)
y_true, y_pred = y_test, knn_gs.predict(X_test)
print(classification_report(y_true, y_pred))