#!usr/bin/env python3

import os
import numpy as np


from nltk.corpus import stopwords
from nltk import TweetTokenizer
from nltk.stem import SnowballStemmer
from string import punctuation
import string
import re

#lexicon word and its classification
class word:
        def __init__(self, word, anger, fear, joy, sadness):
                self.word = word
                self.anger = anger
                self.fear = fear
                self.joy = joy
                self.sadness = sadness

        def print(self):
                print("word " + self.word + " Anger" + str(self.anger), " Fear" +
                      str(self.fear) + " Joy" + str(self.joy) + " Sadness" + str(self.sadness))

#Define weigth for words and emojis
WEIGTH_WORD = 1.0
WEIGTH_EMOJI = 0.2

#Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.
def clean_text(text):
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\w*\f\w*', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*]\)', '', text)
    text = text.lower()
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\t', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text


def load_words():
        direc = "lexico/"
        file = direc + 'palabras_clasficacion.txt'
        clasification = []
        fp = open(file, "r")
        lines = fp.readlines()[1:]
        for x in lines:
                l = x.split('\t')
                # [word, anger, fear, joy, sadness]
                if (l[4] != '0' or l[7] != '0' or l[8] != '0' or l[9] != '0'):
                        s = stemmer.stem(l[0])
                        clasification.append(word(s, int(l[4]), int(l[7]), int(l[8]), int(l[9])))
        fp.close()
        return clasification


def load_emojis():
        direc = "lexico/"
        file = direc + 'emojis_clasificacion.txt'
        clasification = []
        fp = open(file, "r")
        lines = fp.readlines()[1:]
        for x in lines:
                l = x.split('\t')
                # [word, anger, fear, joy, sadness]
                if (l[4] != '0' or l[7] != '0' or l[8] != '0' or l[9] != '0'):
                        clasification.append(
                            word(l[0], int(l[4]), int(l[7]), int(l[8]), int(l[9])))
        fp.close()
        return clasification


def load_twetts():
        direc = "data_set_eng/"
        files = os.listdir(direc)
        files = [direc + twitt for twitt in files]
        twetts = []
        for a in files:
                fp = open(a, "r")
                lines = fp.readlines()[1:]
                for x in lines:
                        words = ''.join([c for c in x.split('\t')[1] if c not in non_words])
                        words = clean_text(words)
                        tt = TweetTokenizer()
                        twitt = tt.tokenize(words)
                        twetts.append([twitt, x.split('\t')[2]])
        fp.close()

        return twetts

def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
                stemmed.append(stemmer.stem(item))
        return stemmed

def switch_sentiment(arg):
    switcher = {
        0: "anger",
                1: "fear",
                2: "joy",
                3: "sadness"
    }
    return switcher.get(arg, "Err")

if __name__ == "__main__":

    spanish_stopwords = stopwords.words('english')
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    non_words = list(punctuation)
    non_words.extend(['¿', '¡'])
    non_words.extend(map(str, range(10)))

    words = load_words()
    emojis = load_emojis()
    twetts = load_twetts()

    # number of correctly classified tweets
    correct = 0
        #number of misclassified tweets
    incorrect = 0
    #number of words
    cant_words = 0
    # number of words that could not be classified
    not_classified = 0

    for t in twetts:
        ps = stem_tokens(t[0], stemmer)
        #[anger, fear, joy, sadness]
        p_clasification = [0, 0, 0, 0]

        for p in ps:
                cant_words += 1
                c = [x for x in words if x.word == p]
                if c:
                    # If I found the word in the word lexicon I update the weight counters
                    p_clasification[0] += c[0].anger
                    p_clasification[1] += c[0].fear
                    p_clasification[2] += c[0].joy
                    p_clasification[3] += c[0].sadness
                else:
                    c = [x for x in emojis if x.word == p]
                    if c:
                        # If I found the word in the emoji lexicon I update the weight counters
                        p_clasification[0] += c[0].anger 
                        p_clasification[1] += c[0].fear 
                        p_clasification[2] += c[0].joy 
                        p_clasification[3] += c[0].sadness
                    else:
                        not_classified += 1
                # determine the sentiment of the tweet, returning the maximum counter
                indice = np.where(p_clasification == np.amax(p_clasification))[0]
                result = "sadness"
                if p_clasification[0] >= p_clasification[1] and p_clasification[0] > p_clasification[2] and p_clasification[0] > p_clasification[3]:
                        result = "anger"
                elif p_clasification[1] >= p_clasification[2] and p_clasification[1] > p_clasification[3]:
                        result = "fear"
                elif p_clasification[2] >= p_clasification[3]:
                        reultado = "joy"
                
        result = switch_sentiment(indice[0])
        # If the calculated result matches that of the data set
        if (result == t[1]):
                correct += 1
        else:
                incorrect += 1
                
    print("------------------------------------------------------------")
    print(str(len(twetts)) + " tweets analyzed") 
    print("correct: " + str(correct))
    print("incorrect: " + str(incorrect))
    print("------------------------------------------------------------")
    print("Total words: " + str(cant_words))
    print("words not clasified: " + str(not_classified))
    print("------------------------------------------------------------")