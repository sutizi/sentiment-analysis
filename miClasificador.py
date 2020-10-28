#!usr/bin/env python3

import os
import numpy as np


from nltk.corpus import stopwords
from nltk import TweetTokenizer
from nltk.stem import SnowballStemmer
from string import punctuation

#Palabra del lexico y su clasificacion
class Palabra:
	def __init__(self, palabra, anger, fear, joy, sadness):
		self.palabra = palabra
		self.anger = anger
		self.fear = fear
		self.joy = joy
		self.sadness = sadness
	
	def print(self):
		print("Palabra "+ self.palabra + " Anger"+ str(self.anger), " Fear" + str(self.fear) +" Joy" + str(self.joy) + " Sadness" + str(self.sadness))
	

def cargar_palabras():
	direc = "data_set/"
	archivo = direc + 'palabras_clasficacion.txt'
	clasificacion = [] 
	fp = open(archivo, "r")
	lineas = fp.readlines()[1:]
	for x in lineas:
		l = x.split('	')
		# [palabra, anger, fear, joy, sadness]
		if (l[4] != '0' or l[7] != '0' or l[8] != '0' or l[9] != '0'):
			s = stemmer.stem(l[1])
			clasificacion.append(Palabra(s, int(l[4]), int(l[7]), int(l[8]), int(l[9])))
	fp.close()
	clasificacion_emojis = cargar_emojis()
	return clasificacion + clasificacion_emojis

def cargar_emojis():
	direc = "data_set/"
	archivo = direc + 'emojis_clasificacion.txt'
	clasificacion = [] 
	fp = open(archivo, "r")
	lineas = fp.readlines()[1:]
	for x in lineas:
		l = x.split('	')
		# [palabra, anger, fear, joy, sadness]
		if (l[4] != '0' or l[7] != '0' or l[8] != '0' or l[9] != '0'):
			clasificacion.append(Palabra(l[0], int(l[4]), int(l[7]), int(l[8]), int(l[9])))
	fp.close()
	return clasificacion

def cargar_twitts():
	direc = "data_set/"
	files = os.listdir(direc)
	archivos = [direc + twitt for twitt in files]
	twitts = []
	for a in archivos:
		fp = open(a, "r")
		lineas = fp.readlines()[1:]
		for x in lineas:
			palabras = ''.join([c for c in x.split('	')[1] if c not in non_words])
			tt = TweetTokenizer()
			twitt = tt.tokenize(palabras)
			twitts.append([twitt, x.split('	')[2]])
	fp.close()

	return twitts

#Stems

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
	
def switch_sentimiento(arg):
    switcher = {
        0: "anger", 
		1: "fear", 
		2: "joy",
		3: "sadness"
    }
    return switcher.get(arg, "Err")

if __name__ == "__main__":

	palabras = cargar_palabras()
	twitts = cargar_twitts()

	#Cantidad de twitts clasificados correctamente
	correctos = 0
	#Cantidad de twitts clasificados incorrectamente
	incorrectos = 0
	#Cantidad total de palabras
	cant_palabras = 0
	#Cantidad de palabras que no se pudieron clasificar
	no_clasificada = 0

	for t in twitts:
		ps = stem_tokens(t[0], stemmer)

		#Contiene la calsificacion de todas las palabras del twitt
		#[anger, fear, joy, sadness]
		p_clasificacion = [0, 0, 0 , 0]

		for p in ps:
			cant_palabras += 1
			c = [x for x in palabras if x.palabra == p]
			if c:
				#Si encontre la palabra en la lista de palabras actualizo los contadores
				p_clasificacion[0] += c[0].anger
				p_clasificacion[1] += c[0].fear
				p_clasificacion[2] += c[0].joy
				p_clasificacion[3] += c[0].sadness
		
				no_clasificada += 1
			else:
					print(p)
		#Determino el sentimiento del twitt
		#Retornando el maximo contador
		indice = np.where(p_clasificacion == np.amax(p_clasificacion))[0]
		resultado = switch_sentimiento(indice[0])

		#Si el resultado calculado coincide con el del data set
		if (resultado == t[1]):
			correctos += 1
		else:
			incorrectos += 1
	print("------------------------------------------------------------")
	print("De " + str(len(twitts)) + " twitts analizados") 
	print("correctos: " + str(correctos))
	print("incorrectos: " + str(incorrectos))
	print("------------------------------------------------------------")
	print("Total palabras: " + str(cant_palabras))
	print("Palabras no clasificadas: " + str(no_clasificada))
	print("------------------------------------------------------------")