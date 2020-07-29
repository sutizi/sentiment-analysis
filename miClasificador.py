#!usr/bin/env python3

import os

def cargar_archivo():
	direc = "data_set/"
	files = os.listdir(direc)
	print(files)
	archivo = direc + 'palabras_clasficacion.txt'
	clasificacion = [] 
	fp = open(archivo, "r")
	lineas = fp.readlines()[1:]
	for x in lineas:
		l = x.split('	')
		# [palabra, anger, fear, joy, sadness]
		clasificacion.append([l[1], l[4], l[7], l[8], l[9]])
	fp.close()
	return clasificacion
	

if __name__ == "__main__":
	a= cargar_archivo()
	print(a)
