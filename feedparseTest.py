# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:39:24 2017

@author: Jim
"""

import pandas as pd
import numpy as np
import feedparser as fp
import calendar

# get feed sources & URLs
feed_list = pd.DataFrame({'feed_source':['BBCNews', 'Guardian', 'Independent', 'alJazeera'],
                          'feed_url':['http://feeds.bbci.co.uk/news/rss.xml',
                          'https://www.theguardian.com/uk/rss',
                          'http://www.independent.co.uk/rss',
                          'http://www.aljazeera.com/xml/rss/all.xml']})

n = feed_list['feed_url'].count() # number of feeds to be read
maxArticles = 50 # maximum number of articles to read from each feed

for i in range(0, n):
    feed_url = feed_list['feed_url'][i] 
    feed = fp.parse(feed_url)
    
    m = (len(feed.entries))
    if m > maxArticles:
        m = maxArticles # only use maxArticles if less than actual number of articles
    
    articles_i = pd.DataFrame(feed.entries, columns = ['published_parsed', 'title', 'summary', 'link'])
    if i == 0:
        articles_df = articles_i[:m][:]
    else:
        articles_df = pd.concat([articles_df, articles_i[:m][:]])
        
articles_df = articles_df.reset_index(drop=True)
articles_df['published_parsed'] = articles_df['published_parsed'].apply(calendar.timegm) #change to UNIX time
articles_df = articles_df.rename(columns={'published_parsed': 'time'})

print(articles_df.shape)
print(articles_df.head())    




