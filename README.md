# News Article Matching Project
Determining if news articles about the same news story based on similarity of title, summary and publish time.

### readArticles.py
Gather news articles from RSS feeds. Feed titles and URLs can be added in groups as csv files to the "sources/" folder.

Aquired articles must be matched externally, extracting for each article a publishing time, title, summary, story URL, source name and StoryID. The StoryID is a unique identifier for each news story, used to match articles about the same story.

### matchAnalysis.py
The matched data can be analysed for properties that could be suggest if articles are likely or unlikely to be about the same news story.

### train.py
Properties of importance are used to train machine learning algorithms to identify matches between unlabelled data.
