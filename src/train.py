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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

input_location = ""
input_file_name = "labelledArticles.csv"

SAVE_FEATURES = True
LOAD_FEATURES = True
output_location = "output/"
output_features_name = "articlePairFeatures.csv"

DROP_DISTINCT = True  # Don't use article pairs without any word content similarity

n_words = 1000  # Number of words using in bag of words
n_matched_desired = 150  # Maximum number of matched pairs used in the full trainin & test data set

POLY_DEGREE = 4  # Degree of polynomial features to generate

if not LOAD_FEATURES:
    # Load matched article data
    print("Loading data...")
    articles = pd.read_csv(input_location + input_file_name, header=0, sep=",", encoding="ISO-8859-1")
    n_articles = len(articles['title'])

    # Generate bag of words for the article titles and summaries
    articles['words'] = articles['title'] + articles['summary']
    article_vectorizer = CountVectorizer(max_df=0.99, min_df=2, max_features=n_words,
                                         stop_words='english', decode_error='ignore',
                                         analyzer='word')
    freq = (article_vectorizer.fit_transform(articles['words'])).toarray()
    freq = normalize(freq)
    print('\nNumber of words: ' + str(freq.shape[1]))

    # Create features (difference between each article in publish time and word content)
    print("Creating features...")
    feature_vectors = pd.DataFrame(columns=('Article1', 'Article2', 'deltaTime',
                                            'deltaFreq', 'label', 'StoryID'))
    for i in range(n_articles):
        for j in range(i + 1, n_articles):
            sameArticle = 0
            story_id = 0
            if articles['StoryID'][i] == articles['StoryID'][j]:
                sameArticle = 1
                story_id = articles['StoryID'][i]
            feature_vectors = feature_vectors.append({'Article1': i,
                                                      'Article2': j,
                                                      'deltaTime': abs(articles['time'][i] - articles['time'][j]),
                                                      'deltaFreq': np.linalg.norm(np.subtract(freq[i], freq[j])),
                                                      'label': sameArticle,
                                                      'StoryID': story_id}, ignore_index=True)
    if SAVE_FEATURES:
        feature_vectors.to_csv(output_location + output_features_name, sep=",", index_label=False)
        print("Features saved")
else:
    feature_vectors = pd.read_csv(output_location + output_features_name, sep=",")
    print("Features loaded from file")

if DROP_DISTINCT:
    # Drop pairs without any word similarity (2**0.5 is displacement between unit vectors without overlap)
    feature_vectors = feature_vectors[feature_vectors.deltaFreq != 2 ** 0.5]

# Get feature details
n_matched_pairs = feature_vectors['label'].sum()
print('\nMatched paired articles: ' + str(n_matched_pairs))
print('Total paired articles: ' + str(len(feature_vectors['label'])))

# Select a training set
matched_vectors = feature_vectors[feature_vectors.label == 1]
unmatched_vectors = feature_vectors[feature_vectors.label == 0]
propMatchedArticles = 0.5  # ratio of matched to unmatched pairs. Don't set to 0.

if n_matched_desired < n_matched_pairs:
    n_matched_desired = n_matched_pairs

n_unmatched = round(n_matched_pairs / propMatchedArticles)
if n_unmatched > len(unmatched_vectors['label']):
    n_unmatched = len(unmatched_vectors['label'])

matched_vectors = matched_vectors.sample(n_matched_desired, axis=0)
unmatched_vectors = unmatched_vectors.sample(n_unmatched, axis=0)
sample_data = pd.concat([matched_vectors, unmatched_vectors])

pipe = Pipeline([
    ('feature_scale', StandardScaler()),
    ('poly', PolynomialFeatures(POLY_DEGREE)),
    ('log_reg', linear_model.LogisticRegression())
])

X = sample_data[['deltaFreq', 'deltaTime']].as_matrix()
y = sample_data[['label']].as_matrix()

# Split into training and test sets
msk = np.random.rand(len(X)) < 0.75
X_train = X[msk, :]
X_test = X[~msk, :]
y_train = y[msk, :]
y_test = y[~msk, :]

pipe.fit(X_train, y_train)
print("Training set score: " + str(pipe.score(X_train, y_train)))
print("Test set score: " + str(pipe.score(X_test, y_test)))

# Plot Results

# Generate grid for colour plot
n_grid = 2000  # Grid size
x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max() + .1
y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max() + .1

x_grid_size = (x_max - x_min) / n_grid
y_grid_size = (y_max - y_min) / n_grid
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_grid_size),
                     np.arange(y_min, y_max, y_grid_size))

grid = np.transpose(np.vstack((xx.ravel(), yy.ravel())))
Z = pipe.predict(grid)

# Plot the grid
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap='coolwarm')

# Add the test set data points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, cmap='bwr', edgecolor='black')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.show()
