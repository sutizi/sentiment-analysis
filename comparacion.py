
import os
import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#import models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from ClfSwitcher import ClfSwitcher

spanish_stopwords = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish', ignore_stopwords=True)
non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str,range(10)))
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed

def tokenize(text):
    text = ''.join([c for c in text if c not in non_words])
    tokens =  word_tokenize(text)
    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
               stop_words = spanish_stopwords,)


pipeline = Pipeline([
    ('vect', vectorizer),
    ('clf', ClfSwitcher()),
])
# Aqui definimos el espacio de parametros a explorar

parameters = [
    {
    
    #TODO: 
    #Recuperar informacion: porcentaje de certidumbre en cada clasifcador/ disribucion (bayesianos)

    #Support Vector Classification
    #TODO: definir kernel param
    #Param: kernel -> separacion en hiperplanos
    'clf__estimator': [LinearSVC(dual=False)],
    #ignore terms that appear in more than x% of the documents
    'vect__max_df': (0.4, 0.2),
    #ignore terms that appear in less than x% of the documents
    'vect__min_df': (0.005, 0.00009, 0.011),
    'vect__max_features': (None, 500),
    #Ver N-grams para caracteres
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas o bigramas
    'clf__estimator__C': (0.2, 0.5, 0.7),
    'clf__estimator__max_iter': (300,  500),
    },
    {
        'clf__estimator': [MLPClassifier()],
        'clf__estimator__solver': ['lbfgs'],
    },
    {
        'clf__estimator': [MultinomialNB()],
        'clf__estimator__alpha': (0.9, 1),#(1e-2, 1e-3, 1e-1),
    },
    {
        'clf__estimator': [SGDClassifier()],
        'clf__estimator__tol': [1e-4], #Tolerance for stopping criteria
    },
    {
        'clf__estimator': [LogisticRegression()],
        'clf__estimator__penalty': ['none'],
    },
]

def cargar_archivos():
	direc = "data_set/"
	files = os.listdir(direc)
	archivos = [direc + twitt for twitt in files]
	texto = []
	sentimiento = []
	rango = []
	for a in archivos:
		fp = open(a, "r")
		lineas = fp.readlines()[1:]
		for x in lineas:
			texto.append(x.split('	')[1])
			sentimiento.append(x.split('	')[2])
			rango.append(x.split('	')[3])
	fp.close()
	

	return texto, sentimiento, rango


t, s, r = cargar_archivos()

X_train, X_test, y_train, y_test = train_test_split(t, s, test_size=0.2, random_state=1234)

# multiprocessing requires the fork to happen in a __main__ protected block
if __name__ == "__main__":

    #to find better hyperparameters values
    clf = GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='f1_macro', return_train_score=True)
    clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()


means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()
