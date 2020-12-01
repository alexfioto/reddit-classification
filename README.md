# Subreddit Classification

![](./assets/reddit-2.jpg)


## Introduction
Utilizing PushShift's api to scrape Reddit post data, this project aims to classify posts as either from [Machine Learning](https://www.reddit.com/r/MachineLearning/) or from [Artificial Intelligence](https://www.reddit.com/r/ArtificialInteligence/) using data from the respective subreddits.


I utilize the following python libaries:

- **requests**: web scraping
- **matplotlib**: data visualization
- **nltk**: natural language processing
- **numpy**: scientific computing
- **pandas**: data importing, cleaning, and exploratory analysis
- **scikit-learn**: machine learning
- **yellowbrick**: data visualization


## Problem Statement

Trained on data from two similar subreddits, how well can a model classify a Reddit post? This project seeks to successfully classify posts belonging to the Machine Learning and Artificial Intelligence subreddits.

Using accuracy and f-1 score as primary metrics, I will train a Naive-Bayes and Random Forest Classifier using 20,000 rows of data scraped from Reddit.


## Executive Summary

I will be analyzing and modeling data scraped from Reddit. The subreddits to be classified are from the Machine Learning and Artificial Intelligence subreddits. After webscraping and exploratory data analysis, I found many common words between the two subreddits. I preprocessed the text using lemmatization, lowercasing, to be fed into word vectorizers. I used two types of word vectorizers: a Count Vectorizer and a Term Frequency-Inverse Document Frequency vectorizer (or TFIDF). I used two classification models, a Multinomial Naive Bayes and a Random Forest. The Random Forest classifier scored the highest accuracy of 85%. 

UPDATE: Added Neural Network model. Highest accuracy is approximately 83%. 


## Overview

- Importing libraries
- Testing PushShift API and automating
- Initial data inspection
- Data cleaning
- Text Preprocessing and EDA
- Feature creation and selection
- Basic Naive-bayes classification model aided by grid searching
- Basic Random Forest classification model aided by grid searching
- Conclusions and recommendations
 

### Notebook 3.1 Web Scraping

First import the necessary libraries: 
 - requests: web scraping
 - numpy: scientific computing
 - pandas: data analysis
 - time: automation
 
First, I test the pulling data from reddit using PushShift's API. After reviewing the process, a function is created to automate web scraping and data frame creating for each subreddit. I will collect 10,000 datapoints from each subreddit resulting in final data frame of 20,000 rows.

After data collection, I concatenated and cleaned the data. Lastly, I created one feature labeled, `all_text` that includes all of the text from `title` and `self_text`.


### Notebook 3.2 Text Preprocessing and EDA

- nltk: natural language processing
- matplotlib: data visualization
- sklearn: word vectorization

First, I create two functions: one that lemmatizes text and another that will stem text. Next, I create two new columns in my dataframe: one is `all_text` lemmatized and another is `all_text` stemmed. 

Next, I investigate words that are common in both subreddits. Words that are found in both subreddits frequently are added to a list of potential stop words. I found that excluding these words hurt model performance.
 

### Notebook 3.3 Naive Bayes Classification Model

- pandas: data analysis
- matplotlib: data visualization
- yellowbrick: data visualization
- sklearn: machine learning

Created X, explanatory variables, and y, target variable. 

First, I instantiated, fit, scored and visualized a basic Naive-Bayes Classifier. I found that it did a decent job at classifying the two subreddits out of the gate. I used scikit-learn's grid search to search for the best hyperparameters. 

Ultimately, I found that an alpha of 250 and 'english' stopwords in a TfidfVectorizer produced the best accuracy score of around 82%.

The notebook includes more details of my findings and visualizations.

### Notebook 3.4 Random Forest Classifier

- pandas: data analysis
- matplotlib: data visualization
- yellowbrick: data visualization
- sklearn: machine learning

This notebook follows the same process outlined in notebook 3.3 using a Random Forest Classifier. 

The Random Forest Classifier takes a quite a long time to grid search and I am continuously updating and improving my model. Currently, my best search is: 

{'randomforestclassifier__criterion': 'entropy',
 'randomforestclassifier__n_estimators': 150,
 'tfidfvectorizer__max_features': 9000,
 'tfidfvectorizer__min_df': 1,
 'tfidfvectorizer__ngram_range': (1, 1),
 'tfidfvectorizer__stop_words': 'english'}

This grid search resulted a test accuracy of 85%

The bottom of the notebook includes some of my attempts at other classifier models.
 
## Conclusion

Accuracy will be used as the metric to determine best model since this project simply aims to correctly classify subreddits rather than optimize for precision or recall. 

Between the two models, I found that the Random Forest Classifier performed best. Both models seem to misclassify Machine Learning posts as AI posts more frequently than the opposite. 

These are the features (or words) that had the most impact on the performance model in order of importance:


|Feature Name|
|------|
|removed|
|ai|
|artificial|
|intelligence|
|learning|
|machine|
|model|

I found that the word "removed" was found more frequently in the Artifical Intelligence subreddit indicating that more users removed posts than in the Machine Learning subreddit. 

I also discovered that lemmatizing the text produced better results than stemming or no preprocessing. 

This classification problem was relatively difficult as the two topics are very similar and use much of the same vocabulary. It seemed that both of my models seemed to perform better with more information. The more I suppressed the data input the worse the model performed.

Looking forward, I hope to continue to hone NLP preprocessing skills to achieve higher accuracy scores. 

### Additional Classifiers

I ended up trying HistGradientBoost, AdaBoost and SVC and Gradient Boost Classifiers. None of these worked well out of the box. I found that a basic Logistic Regression worked almost as well as my best Random Forest Classifier. 

Thank you for your interest!


## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**title**|*string*|processed_posts|Title of the subreddit|
|**subreddit**|*int*|processed_posts|Binary. Machine Learning: 1, Artificial Intelligence: 0|
|**selftext**|*string*|processed_posts|Text of the post|
|**permalink**|*string*|processed_posts|Link to post|
|**author**|*string*|processed_posts|Author of post|
|**created_utc**|*int*|processed_posts|Epoch time of post creation|
|**media_only**|*boolean*|processed_posts|True if post is only media|
|**all_text**|*string*|processed_posts|Combined title and selftext|
|**stem_all_text**|*string*|processed_posts|Stemmed all_text|
|**lem_all_text**|*string*|processed_posts|Lemmatized all_text|
