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


class CleanProcess:
    def __init__(self, tweet):
        self.tweet = tweet

    def round1(self, tweet):
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

        re_t, find_m, find_h = None, None, None
        if tweet.truncated:
            tweet_clean = re.sub('(\\\\[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet.__dict__
                                 ['_json']['extended_tweet']['full_text'])
        else:
            tweet_clean = re.sub(
                '(\\\\[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet.text)

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
        # rempve bitly links
        tweet_clean = re.sub(r'bit.ly/\S+', '', tweet_clean)
        tweet_clean = tweet_clean.strip('[link]')  # remove [links]

        '''Remove accented chars'''
        tweet_clean = unicodedata.normalize('NFKD', tweet_clean).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')

        '''Expanded text'''
        tweet_clean = expand_contractions(tweet_clean)

        return [tweet_clean, re_t, find_m, find_h, emos]
