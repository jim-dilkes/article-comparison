# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:22:29 2017

@author: Jim

Reformat the labelled data exported from an Access database
"""

import pandas as pd
from nltk.tokenize import word_tokenize

def toAscii(inputString):
    return "".join(char if ord(char) < 128 else '' for char in inputString)

print("Loading data...")
articles = pd.read_csv('labelledArticle.csv', header = 0, sep="\t", encoding = "ISO-8859-1")
articles = articles.drop('ArticleID', axis=1)
n_articles = len(articles['title'])

articles['title'] = articles['title'].apply(lambda x: x.replace(',', ' ').replace('[', '').replace(']', '').replace("'", ''))
articles['summary'] = articles['summary'].apply(lambda x: x.replace(',', ' ').replace('[', '').replace(']', '').replace("'", ''))
articles['title'] = articles['title'].apply(toAscii)
articles['summary'] = articles['summary'].apply(toAscii)

with open("dictionary.txt") as inputString:
    dictionaryList = inputString.read().splitlines()

#words_series = articles['summary'] + ' ' + articles['title']
articles['summary'] = articles['summary'].apply(word_tokenize)
articles['title'] = articles['title'].apply(word_tokenize) 

print(type(articles['title']))
#Compare to dict and detokenize
def dictAndDetokenize(words_list):
    matches = [c for c in words_list if c in dictionaryList]
    return " ".join(matches)

articles['summary'] = articles['summary'].apply(dictAndDetokenize)
articles['title'] = articles['title'].apply(dictAndDetokenize) 

print(articles.head())

articles.to_csv('reFormatArticles.csv', sep='\t')