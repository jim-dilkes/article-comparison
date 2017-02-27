# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:07:11 2017

@author: Jim
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math


def toAscii(inputString):
    return "".join(char if ord(char) < 128 else '' for char in inputString)

print("Loading data...")
articles = pd.read_csv('labelledArticle.csv', header = 0, sep="\t", encoding = "ISO-8859-1")

n_articles = len(articles['title'])
articles['words'] = articles['title'] + articles['summary']

article_vectorizer = CountVectorizer(max_df=0.99, min_df=2, max_features = 400,
                                      stop_words='english', decode_error='ignore',
                                      analyzer = 'word')
X = article_vectorizer.fit_transform(articles['words'])
freq = X.toarray()

print('Number of features: ' + str(X.shape[1]))
print(article_vectorizer.get_feature_names())

#Regularise each frequency vector
print()
print(type(freq))
print(freq.shape)   
freq = normalize(freq)
#Create features (diff between stories)

featureVectors = pd.DataFrame(columns=('Article1', 'Article2', 'deltaTime',
                                    'deltaFreq', 'label', 'StoryID'))
for i in range(n_articles):
        for j in range(i+1, n_articles):
            sameArticle = 0;
            story_id=0;
            if articles['StoryID'][i] == articles['StoryID'][j]:
                sameArticle = 1;
                story_id = articles['StoryID'][i]
            featureVectors = featureVectors.append({'Article1': i,
             'Article2': j,
             'deltaTime': abs(articles['time'][i] - articles['time'][j]),
             'deltaFreq': np.linalg.norm(np.subtract(freq[i],freq[j])),
             'label': sameArticle,
             'StoryID': story_id }, ignore_index=True)
    
matched_pairs = featureVectors['label'].sum()
total_pairs = len(featureVectors['label'])
print('\nMatched paired articles: ' + str(matched_pairs))
print('Total paired articles: ' + str(total_pairs))

matchedVectors = featureVectors[featureVectors.label == 1]
unmatchedVectors = featureVectors[featureVectors.label == 0]

propMatchedArticles = 1/2 #ratio of matched to unmatched pairs. Don't set to 0.

totUnmatched = round(matched_pairs/propMatchedArticles)
if totUnmatched > len(unmatchedVectors['label']): totUnmatched = unmatchedVectors
selectUnmatched = unmatchedVectors.sample(totUnmatched, axis = 0)

frames = [matchedVectors, selectUnmatched]
sampleVectors = pd.concat(frames)

#Present as similarity
sampleVectors['deltaFreq'] = sampleVectors['deltaFreq'].subtract(math.sqrt(2)).multiply(-1)
print('\nFeature vectors sample:')
print(sampleVectors.sample(10))

plt.scatter(sampleVectors['deltaTime'], sampleVectors['deltaFreq'], c=sampleVectors['label'], cmap='bwr')







