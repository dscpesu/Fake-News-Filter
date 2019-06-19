# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('fakenews dataset.tsv', delimiter = '\t',encoding = "ISO-8859-1", quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
bow=[]
for i in range(0,69):
    review=re.sub('[^a-zA-Z]', ' ', dataset['Article'][i])
    review=review.lower()#converts all characters to lowercase
    review=review.split()#splits the sentence into a list
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]# removal of stopwords
    review=' '.join(review)#converting the list back into a sentence
    corpus.append(review)#creating a list of sentences
    bow.append(review.split(" "))#creating a list of words in each sentences and storing it in a list

#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("done")