#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import re
import string
import sys
import argparse
import json
import pandas as pd
import glob
import numpy as np

import emoji
import datetime
import regex
from collections import Counter
from contractions import CONTRACTION_MAP
import unicodedata

df_tweets = pd.read_csv('WorkFile.csv')
df_tweets['text'][0]


# df_tweets = df_tweets[df_tweets['truncated'] == False]
df_tweets.drop(['hashtags', 'urls', 'mentions',
                'truncated'], axis=1, inplace=True)
df_tweets.shape


def extract_emo_from_text(text):
    emoji_list = []
    tokens = regex.findall(r'\X', text)
    # print(tokens)
    for word in tokens:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)
    return emoji_list


def extract_emojis(tweets):
    emo = list()
    for tweet in tweets:
        tweet_emos = extract_emo_from_text(tweet)
        emo += tweet_emos
    emo = Counter(emo)
    return emo


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(
            match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def find_retweeted_clean(tweet):
    re_t, find_m, find_h = None, None, None
    tweet_clean = re.sub('(\\\\[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet.text)
    emos = extract_emojis(tweet_clean)
    for emo in emos:
        tweet_clean = re.sub(emo, ' ', tweet_clean)

    '''This function will extract the twitter handles of retweed people'''
    re_t = re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet_clean)
    tweet_clean = re.sub(':', '', tweet_clean)

    if re_t:
        tweet_clean = re.sub('RT', '', tweet_clean)
        tweet_clean = re.sub(re_t[0], '', tweet_clean)
        pass
    else:
        re_t = None
        tweet_clean = tweet.text

    '''This function will extract the twitter handles of people mentioned in the tweet'''
    find_m = re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet_clean)

    if find_m:
        for find in find_m:
            tweet_clean = re.sub(find, ' ', tweet_clean)
    else:
        find_m = None
        tweet_clean = tweet_clean

    '''This function will extract hashtags'''
    find_h = re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet_clean)
    if find_h:
        for find in find_h:
            tweet_clean = re.sub(find, ' ', tweet_clean)
    else:
        find_h = None
        tweet_clean = tweet_clean

    '''Takes a string and removes web links from it'''
    tweet_clean = re.sub(r'http\S+', '', tweet_clean)  # remove http links
    tweet_clean = re.sub(r'bit.ly/\S+', '', tweet_clean)  # rempve bitly links
    tweet_clean = tweet_clean.strip('[link]')  # remove [links]

    '''Remove accented chars'''
    tweet_clean = unicodedata.normalize('NFKD', tweet_clean).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    '''Expanded text'''
    tweet_clean = expand_contractions(tweet_clean)

    return tweet_clean, re_t, find_m, find_h, emos


df_tweets['textClean'], df_tweets['retweet'], df_tweets['mentions'], df_tweets['hashtags'], df_tweets['pos_emos'] = zip(
    *df_tweets.apply(find_retweeted_clean, axis=1))


df_tweets.head()


my_stopwords = nltk.corpus.stopwords.words('english')
my_stopwords.extend(['amp', 'u', 'pleasse', 'one'])
word_rooter = nltk.stem.WordNetLemmatizer().lemmatize
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@ー'

# cleaning master function


def clean_tweet(tweet, bigrams=False):
    tweet = tweet.lower()  # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                        if word not in my_stopwords]  # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list]  # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                             for i in range(len(tweet_token_list)-1)]
    if len(tweet_token_list) < 2:
        tweet = None
    else:
        tweet = ' '.join(tweet_token_list)
    return tweet


# In[11]:


df_tweets['clean_tweet'] = df_tweets.textClean.apply(clean_tweet)
df_tweets.shape


# In[12]:


df_tweets = df_tweets[df_tweets['clean_tweet'].notna()]
df_tweets.shape


# In[13]:


def splitter_text(text):
    return re.split(' ', text)


word_list = []
for word in df_tweets['clean_tweet'].apply(splitter_text):
    for w in word:
        word_list.append(w)

word_list = list(filter(lambda x: x != "", word_list))
word_list_counter = Counter(word_list).most_common()
len(word_list_counter)


# In[14]:


add_stop_words = [word for word, count in word_list_counter if count > 300]
len(add_stop_words)


# In[15]:


# Add new stop words


def stopwords_round2(text):
    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

    tweet_token_list = [word for word in text.split(' ')
                        if word not in stop_words]  # remove stopwords
    tweet = ' '.join(tweet_token_list)
    return tweet


df_tweets['clean_tweetR2'] = df_tweets.clean_tweet.apply(clean_tweet)
df_tweets.head()


# In[16]:


df_tweets.to_csv('tweets-after-cleaned.csv')
type(df_tweets['clean_tweetR2'][0])


# In[17]:


def pol(x): return TextBlob(x).sentiment.polarity


def sub(x): return TextBlob(x).sentiment.subjectivity


df_tweets_s = pd.DataFrame()
df_tweets_s['text'] = df_tweets['textClean']
df_tweets_s['polarity'] = df_tweets['textClean'].apply(pol)
df_tweets_s['subjectivity'] = df_tweets['textClean'].apply(sub)
df_tweets_s


# In[18]:


df_tweets_s.replace(0.0, np.nan, inplace=True)
df_tweets_s.dropna(subset=['polarity', 'subjectivity'],
                   how='any', inplace=True)
df_tweets_s.replace(np.nan, 0, inplace=True)
df_tweets_s.to_csv('sentiment-analysis-data.csv')
df_tweets_s_p = df_tweets_s.sample(5000)


# In[30]:


plt.rcParams['figure.figsize'] = [10, 8]

for index, ind in enumerate(df_tweets_s_p.index):
    x = df_tweets_s_p.polarity.loc[ind]
    y = df_tweets_s_p.subjectivity.loc[ind]
    if x >= 0.05:
        if y >= 0.5:
            plt.scatter(x, y, color='blue')
        else:
            plt.scatter(x, y, color='gray')
    else:
        if y >= 0.5:
            plt.scatter(x, y, color='green')
        else:
            plt.scatter(x, y, color='red')
#     plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.01, .12)

plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


# In[20]:


df_tweets.head()


# In[21]:


# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(
    max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# apply transformation
tf = vectorizer.fit_transform(df_tweets['clean_tweet']).toarray()

# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()
tf_feature_names


# In[22]:


tf.shape
data_saver = pd.DataFrame(tf)
data_saver.to_csv('topic-modeling-data.csv')


# In[23]:


number_of_topics = 4

model = LatentDirichletAllocation(
    n_components=number_of_topics, random_state=0)


# In[24]:


model.fit(tf)


# In[25]:


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(feature_names[i])
                                                      for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i])
                                                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# In[26]:


no_top_words = 4
display_topics(model, tf_feature_names, no_top_words)


# In[27]:


tf_pre = vectorizer.fit_transform(df_tweets['clean_tweet'][0:10]).toarray()
# z_labels = model.predict(tf_pre)
tf_pre


# In[ ]:

