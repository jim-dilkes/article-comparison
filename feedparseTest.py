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

print('Checking feeds:')
print(feed_list)

n = feed_list['feed_url'].count() # number of feeds to be read
maxArticles = 15 # maximum number of articles that can be read from each feed

for i in range(0, n):
    feed_url = feed_list['feed_url'][i] 
    feed = fp.parse(feed_url)
    
    m = (len(feed.entries))
    if m > maxArticles:
        m = maxArticles # only use maxArticles if less than actual number of articles
    
    articles_i = pd.DataFrame(feed.entries, columns = ['published_parsed', 'title', 'summary', 'link']) # put selected columns from feed.entries into a DataFrame
    articles_i['source'] = pd.Series(feed_list['feed_source'][i], index = range(0, m))
    print(feed_list['feed_source'][i] + ': ' + str(len(articles_i)))
    if i == 0:
        articles_df = articles_i[:m][:]
    else:
        articles_df = pd.concat([articles_df, articles_i[:m][:]])
        
articles_df = articles_df.reset_index(drop=True)
articles_df = articles_df.rename(columns={'published_parsed': 'time'})
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

#Tokenize words
articles_df['summary'] = articles_df['summary'].apply(word_tokenize)
articles_df['title'] = articles_df['title'].apply(word_tokenize)

#Replace all numbers with the word 'number' (probably a much better way to do this)
def containsNumbers(inputString):
    return any(char.isdigit() for char in inputString)
for i in range(0, len(articles_df['summary'])):
        for j in range(0, len(articles_df['summary'][i])):
            if containsNumbers(articles_df['summary'][i][j]):
                articles_df['summary'][i][j] = 'number'
                
for i in range(0, len(articles_df['title'])):
        for j in range(0, len(articles_df['title'][i])):
            if containsNumbers(articles_df['title'][i][j]):
                articles_df['title'][i][j] = 'number'
                                          
print(articles_df.shape)
print(articles_df.head())

articles_df.to_csv('sampleArticles.csv', sep='\t')



