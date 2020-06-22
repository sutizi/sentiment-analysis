#!usr/bin/env python3

import os
import pickle as c
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def guardar(clf, name):
    with open(name, 'wb') as fp:
            c.dump(clf, fp)
	    print "El clasificador fue guardado"
		
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


vectorizer = CountVectorizer(analyzer = 'word', lowercase = False,)

features = vectorizer.fit_transform(t)

features_nd = features.toarray()


f_train, f_test, l_train, l_test = train_test_split(features_nd, s,  test_size = 0.2, random_state=1234)

clf = MultinomialNB()

clf = clf.fit(X=f_train, y=l_train)

preds = clf.predict(f_test)

print "ACCURACY:", accuracy_score(l_test, preds)

guardar(clf, "clasificador.mdl")
