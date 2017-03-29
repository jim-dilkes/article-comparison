# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:57:00 2017

@author: Jim
"""

import pandas as pd
import matplotlib.pyplot as plt

print("Loading data...")
articles = pd.read_csv('output/articlePairFeatures.csv', header=0, sep=",", encoding="ISO-8859-1")

# Show distribution of word content difference between paris of articles
diff_articles = articles[articles.deltaFreq != 2 ** 0.5]  # Remove articles that have no key words in common
same = diff_articles[articles.label == True]
different = diff_articles[articles.label == False]

plt.hist([same.deltaFreq, different.deltaFreq],
         alpha=0.60,
         bins=80,
         range=(0.55, 1.41421356237),
         normed=1,
         histtype='stepfilled',
         label=["Same story", "Different stories"])
plt.legend(loc="upper left")
plt.title("Article content similarity for the same and different news stories")
plt.xlabel("Pair of article's word content difference (pair's dissimilarity)")
plt.ylabel("Normalised frequency")

# Show distribution of publish time difference between paris of articles
same = articles[articles.label == True]
different = articles[articles.label == False]

plt.figure()
plt.hist([same.deltaTime, different.deltaTime],
         alpha=0.60,
         bins=30,
         range=(0, 86400),
         normed=1,
         histtype='stepfilled',
         label=["Same story", "Different stories"])
plt.legend(loc="upper right")
plt.title("Article publish time difference for \nthe same and different news stories in first 24hrs")
plt.xlabel("Pair of article's publish time difference")
plt.ylabel("Normalised Frequency")

plt.show()
