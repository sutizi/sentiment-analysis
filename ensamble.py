import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
import re
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import LinearSVC

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

#Remove spanish stopwords from text
def remove_stopwords(text):
  stop_words = set(stopwords.words('spanish'))
  text = text.apply(lambda x: ' '.join(
      term for term in x.split() if term not in stop_words))
  return text

#Creating Bag of Words for the messages data
def count_words(text):
    all_words = []
    word_counts = []
    for word in text:
        words = word_tokenize(word)
        for w in words:
            all_words.append(w)
    word_counts = Counter(all_words)
    return word_counts

# The find_features function will determine which of the 1500 word features are contained in the review.
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in w_features:
        features[word] = (word in words)

    return features

#Read the dataset
data = pd.read_csv('data.csv', header=None, skiprows=1)
tweets = data[0]
Y = data[1]

tweets = tweets.apply(clean_text)

#lemmatizing using wordnet lemmatizer
lemmatizer = WordNetLemmatizer()
tweets = tweets.apply(lambda x: ' '.join(lemmatizer.lemmatize(term) for term in x.split()))

word_counts = count_words(tweets)

#Using the 1000 most common words as features.
w_features = list(word_counts.keys())[:1000]


features = list(zip(tweets, Y))
seed = 1
np.random.seed = seed
np.random.shuffle(features)
feature_set=[]
for (x,y) in features:
  feature_set.append((find_features(x), y))

# spliting feature set into training data and testing datas
train_data, test_data = model_selection.train_test_split(feature_set, test_size = 0.2, random_state=seed)

# Defining all the models to train

#OPTION 1
model_classifier = [ KNeighborsClassifier(weights= 'distance'), DecisionTreeClassifier(), SGDClassifier(max_iter = 100), MultinomialNB(), SVC(kernel = 'linear')]
model_name = ["K Nearest Neighbors", "Decision Tree", "SGD Classifier", "Naive Bayes", "SVM Linear"]

#OPTION 2
#model_classifier = [RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0), LinearSVC(C=0.2, dual=False, max_iter=300),
#SGDClassifier(max_iter = 100), MultinomialNB(), SVC(kernel = 'linear')]
#model_name = ["Random forest", "LinearSVC", "SGDClassifier",  "Naive Bayes", "SVM Linear"]

all_models = list(zip(model_name, model_classifier))

for model_name, model_classifier in all_models:
    nltk_model = SklearnClassifier(model_classifier)
    nltk_model.train(train_data)
    accuracy = nltk.classify.accuracy(nltk_model, test_data)*100
    print("{} Accuracy: {}".format(model_name, accuracy))

# Voting classifier
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = all_models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(train_data)
accuracy = nltk.classify.accuracy(nltk_ensemble, test_data)*100
print("Voting Classifier Accuracy: {}".format(accuracy))

txtfeatures, labels = zip(*test_data)
prediction = nltk_ensemble.classify_many(txtfeatures)

# print a classification report
print(classification_report(labels, prediction))
confusionmatrix= pd.DataFrame(confusion_matrix(labels, prediction), index = [['actual', 'actual', 'actual', 'actual'], ['0', '1', '2', '3']], columns = [['predicted','predicted','predicted', 'predicted'], ['0', '1', '2', '3']])
print(confusionmatrix)