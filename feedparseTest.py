# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:39:24 2017

@author: Jim
"""

import pandas as pd
import feedparser as fp
import calendar
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize

#Load sources &URLs from file
sources_file_name = 'ukNewsSources.csv'
feed_list = pd.read_csv(sources_file_name, header=0)

print('Accessing feeds:')
print(feed_list)

n_feeds = feed_list['feed_url'].count() # number of feeds to be read
maxArticles = 20 # maximum number of articles that can be read from each feed

articles_df =  pd.DataFrame(columns=('published_parsed', 'title', 'summary', 'link', 'source'))

print('\nNumber of available articles:' )
for i in range(0, n_feeds):
    feed_url = feed_list['feed_url'][i] 
    feed = fp.parse(feed_url)
    n_feed_articles = maxArticles if maxArticles < (len(feed.entries)) else len(feed.entries)
    
    #put selected columns from feed.entries into a DataFrame
    articles_i = pd.DataFrame(feed.entries, columns = ['published_parsed', 'title', 'summary', 'link']) #
    articles_i['source'] = pd.Series(feed_list['feed_source'][i], index = range(n_feed_articles))
    print(feed_list['feed_source'][i] + ': ' + str(len(articles_i)))
    articles_df = pd.concat([articles_df, articles_i[:n_feed_articles][:]])
    print()
        
articles_df = articles_df.rename(columns={'published_parsed': 'time'})
articles_df = articles_df.dropna()
articles_df = articles_df.reset_index(drop=True)
articles_df['time'] = articles_df['time'].apply(calendar.timegm) #change to UNIX time          
  
### Text Preprocessing  ###

articles_df['summary'] = articles_df['summary'].apply(lambda x: BeautifulSoup(x, "lxml").text) 
articles_df['title'] = articles_df['title'].apply(lambda x: BeautifulSoup(x, "lxml").text)

#All chars to lowercase
articles_df['summary'] = articles_df['summary'].apply(lambda x: x.lower())
articles_df['title'] = articles_df['title'].apply(lambda x: x.lower())

#Replace '.' and ',' with no character so that numbers are contained as a single element when tokenized
#Also remove bullet points, apostrophes, single quotes
articles_df['summary'] = articles_df['summary'].apply(lambda x: x.replace(',', '').replace('.', '').replace('•', '').replace("'", '').replace('‘', '').replace('’', ''))
articles_df['title'] = articles_df['title'].apply(lambda x: x.replace(',', '').replace('.', '').replace('•', '').replace("'", '').replace('‘', '').replace('’', ''))

#Replace other punctuation with ' '
puncTable = str.maketrans(string.punctuation, ' '*len(string.punctuation))
articles_df['summary'] = articles_df['summary'].apply(lambda x: x.translate(puncTable))
articles_df['title'] = articles_df['title'].apply(lambda x: x.translate(puncTable)) 

#Replace currency symbols with words
articles_df['summary'] = articles_df['summary'].apply(lambda x: x.replace('£', ' pound ').replace('$', ' dollar ').replace('€', ' euro '))
articles_df['title'] = articles_df['title'].apply(lambda x: x.replace('£', ' pound ').replace('$', ' dollar ').replace('€', ' euro '))

def toAscii(inputString):
    return "".join(char if ord(char) < 128 else '' for char in inputString)

articles_df['title'] = articles_df['title'].apply(toAscii)
articles_df['summary'] = articles_df['summary'].apply(toAscii)

#Tokenize words for analysing individual words
articles_df['summary'] = articles_df['summary'].apply(word_tokenize)
articles_df['title'] = articles_df['title'].apply(word_tokenize)
                
with open("dictionary.txt") as inputString:
    dictionaryList = inputString.read().splitlines()
    
def dictAndDetokenize(words_list):
    return " ".join([c for c in words_list if c in dictionaryList])

def handleNumbers(words_list):
    return [('number' if any(char.isdigit() for char in x) else x) for x in words_list]

articles_df['title'] = articles_df['title'].apply(handleNumbers)
articles_df['summary'] = articles_df['summary'].apply(handleNumbers)
print('\nComparing to dictionary and detokenizing...\n')
articles_df['title'] = articles_df['title'].apply(dictAndDetokenize)
articles_df['summary'] = articles_df['summary'].apply(dictAndDetokenize)
                                                          
print(articles_df.shape)
print(articles_df.head())
#Export for story matching
articles_df.to_csv('sampleArticles_' + sources_file_name, sep='\t')



