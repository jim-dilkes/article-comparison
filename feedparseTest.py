# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:39:24 2017

@author: Jim
"""

import feedparser as fp
import calendar

d = fp.parse('http://feeds.bbci.co.uk/news/rss.xml')

print(d.feed.title)
print(d.feed.description)
n = (len(d.entries))
 
for i in range(0, n):
    print(calendar.timegm(d.entries[i].published_parsed))
    print(d.entries[i].published + ': '+d.entries[i].title +'\n' + d.entries[i].description+ '\n')