# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:07:11 2017

@author: Jim
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn import linear_model

#Load article data

print("Loading data...")
articles = pd.read_csv('labelledArticle.csv', header = 0, sep="\t", encoding = "ISO-8859-1")
n_articles = len(articles['title'])

#Generate bag of words for the article titles and summaries

articles['words'] = articles['title'] + articles['summary']
article_vectorizer = CountVectorizer(max_df=0.99, min_df=2, max_features = 400,
                                      stop_words='english', decode_error='ignore',
                                      analyzer = 'word')
freq = (article_vectorizer.fit_transform(articles['words'])).toarray()

print('\nNumber of words: ' + str(freq.shape[1]))
print(article_vectorizer.get_feature_names())

freq = normalize(freq) 

#Create features (diff between each article in time and word content)

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

#Select a training set (still need to split into training & test set)

matchedVectors = featureVectors[featureVectors.label == 1]
unmatchedVectors = featureVectors[featureVectors.label == 0]

propMatchedArticles = 1/2 #ratio of matched to unmatched pairs. Don't set to 0.

totUnmatched = round(matched_pairs/propMatchedArticles)
if totUnmatched > len(unmatchedVectors['label']): totUnmatched = unmatchedVectors

selectUnmatched = unmatchedVectors.sample(totUnmatched, axis = 0)

frames = [matchedVectors, selectUnmatched]
sampleVectors = pd.concat(frames)

print('\nTraining set sample:')
print(sampleVectors.sample(10))

#Feature Scaling

sampleVectors['deltaTime'] = sampleVectors['deltaTime'].subtract(sampleVectors['deltaTime'].mean())
sampleVectors['deltaTime'] = sampleVectors['deltaTime'].multiply(1/(sampleVectors['deltaTime'].max() - sampleVectors['deltaTime'].min()))

sampleVectors['deltaFreq'] = sampleVectors['deltaFreq'].subtract(sampleVectors['deltaFreq'].mean())
sampleVectors['deltaFreq'] = sampleVectors['deltaFreq'].multiply(1/(sampleVectors['deltaFreq'].max() - sampleVectors['deltaFreq'].min()))


#Generate polynomial features

sampleVectors['deltaTime_2'] = sampleVectors['deltaTime'].apply(lambda x: x ** 2)
sampleVectors['deltaFreq_2'] = sampleVectors['deltaFreq'].apply(lambda x: x ** 2)
sampleVectors['deltaTime_3'] = sampleVectors['deltaTime'].apply(lambda x: x ** 3)
sampleVectors['deltaFreq_3'] = sampleVectors['deltaFreq'].apply(lambda x: x ** 3)

#Fit training set to a logistic regression

X = pd.DataFrame(sampleVectors, columns = ['deltaTime', 
                                           'deltaFreq', 
                                           'deltaTime_2', 
                                           'deltaFreq_2',
                                           'deltaTime_3', 
                                           'deltaFreq_3'
                                           ]).as_matrix()
y = sampleVectors['label']
logreg = linear_model.LogisticRegression()
logreg.fit(X, y)
print('\nTraining set score: ' + str(logreg.score(X, y)))

print('\nintercept_ '  + str(logreg.intercept_.shape) + ':\n' + str(logreg.intercept_ ))
print('\ncoef_ ' + str(logreg.coef_.shape) + ':\n' + str(logreg.coef_))

#Plot Results

#Generate grid for colour plot
h = .005
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .05, X[:, 1].max() + .05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h/20))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel() ** 2, yy.ravel() **2, xx.ravel()**3, yy.ravel()**3])

#Put the result into a colour plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap='coolwarm')

#Add the training set data points
plt.scatter(sampleVectors['deltaTime'], sampleVectors['deltaFreq'], c=sampleVectors['label'], cmap='bwr', edgecolor='black')
plt.show()


