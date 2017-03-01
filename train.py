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


n_words = 1000


#Load article data

print("Loading data...")
articles = pd.read_csv('labelledArticles.csv', header = 0, sep=",", encoding = "ISO-8859-1")
n_articles = len(articles['title'])

#Generate bag of words for the article titles and summaries

articles['words'] = articles['title'] + articles['summary']
article_vectorizer = CountVectorizer(max_df=0.99, min_df=2, max_features = n_words,
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
    
    

n_matched_pairs = featureVectors['label'].sum()
n_total_pairs = len(featureVectors['label'])
print('\nMatched paired articles: ' + str(n_matched_pairs))
print('Total paired articles: ' + str(n_total_pairs))

#Select a training set (still need to split into training & test set)

matchedVectors = featureVectors[featureVectors.label == 1]
unmatchedVectors = featureVectors[featureVectors.label == 0]

propMatchedArticles = 1/2 #ratio of matched to unmatched pairs. Don't set to 0.
n_matched_desired = 150

if n_matched_desired < n_matched_pairs: n_matched_desired = n_matched_pairs

n_unmatched = round(n_matched_pairs/propMatchedArticles)
if n_unmatched > len(unmatchedVectors['label']): n_unmatched = len(unmatchedVectors['label'])

matchedVectors = matchedVectors.sample(n_matched_desired, axis = 0)
unmatchedVectors = unmatchedVectors.sample(n_unmatched, axis = 0)

frames = [matchedVectors, unmatchedVectors]
sampleData = pd.concat(frames)

print('\nTraining set sample:')
print(sampleData.sample(10))

#Feature Scaling

sampleData['deltaTime'] = sampleData['deltaTime'].subtract(sampleData['deltaTime'].mean())
sampleData['deltaTime'] = sampleData['deltaTime'].multiply(1/(sampleData['deltaTime'].max() - sampleData['deltaTime'].min()))

sampleData['deltaFreq'] = sampleData['deltaFreq'].subtract(sampleData['deltaFreq'].mean())
sampleData['deltaFreq'] = sampleData['deltaFreq'].multiply(1/(sampleData['deltaFreq'].max() - sampleData['deltaFreq'].min()))

#Generate polynomial features

sampleData['deltaTime_2'] = sampleData['deltaTime'].apply(lambda x: x ** 2)
sampleData['deltaFreq_2'] = sampleData['deltaFreq'].apply(lambda x: x ** 2)
sampleData['deltaTime_3'] = sampleData['deltaTime'].apply(lambda x: x ** 3)
sampleData['deltaFreq_3'] = sampleData['deltaFreq'].apply(lambda x: x ** 3)
sampleData['deltaTime_4'] = sampleData['deltaTime'].apply(lambda x: x ** 4)
sampleData['deltaFreq_4'] = sampleData['deltaFreq'].apply(lambda x: x ** 4)

#Split into training and test sets

msk = np.random.rand(len(sampleData))<0.8
trainData = sampleData[msk]
testData = sampleData[~msk]

print(trainData.head())

#Fit training set to a logistic regression

Xtrain = pd.DataFrame(trainData, columns = ['deltaTime', 
                                           'deltaFreq', 
                                           'deltaTime_2', 
                                          'deltaFreq_2',
                                           'deltaTime_3', 
                                          'deltaFreq_3',
                                          'deltaTime_4' 
#                                           'deltaFreq_4'
                                           ]).as_matrix()
    
Xtest = pd.DataFrame(testData, columns = ['deltaTime', 
                                           'deltaFreq', 
                                           'deltaTime_2', 
                                          'deltaFreq_2',
                                           'deltaTime_3', 
                                           'deltaFreq_3',
                                          'deltaTime_4' 
#                                           'deltaFreq_4'
                                           ]).as_matrix()

print(Xtrain.shape)
y = trainData['label']
logreg = linear_model.LogisticRegression()
logreg.fit(Xtrain, y)
print('\nTraining set score: ' + str(logreg.score(Xtrain, y)))
print('\nTest set score: ' + str(logreg.score(Xtest, testData['label'])))


print('\nintercept_ '  + str(logreg.intercept_.shape) + ':\n' + str(logreg.intercept_ ))
print('\ncoef_ ' + str(logreg.coef_.shape) + ':\n' + str(logreg.coef_))

#Plot Results

#Generate grid for colour plot
h = .005
x_min, x_max = Xtrain[:, 0].min() - .1, Xtrain[:, 0].max() + .1
y_min, y_max = Xtrain[:, 1].min() - .05, Xtrain[:, 1].max() + .05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h/20))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel() ** 2, yy.ravel() ** 2, xx.ravel() ** 3, yy.ravel() ** 3, xx.ravel() ** 4])

#Put the result into a colour plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap='coolwarm')

#Add the training set data points
plt.scatter(testData['deltaTime'], testData['deltaFreq'], c=testData['label'], cmap='bwr', edgecolor='black')
plt.show()


