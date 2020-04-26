# imports
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import textwrap

from threading import Thread
from collections import deque
from requests.exceptions import ChunkedEncodingError
from streamprocess import CleanProcess
from live_classifier import Sentiment

import time
import sys
from sys import stdout
import os

import csv
import json
import re


def clean_text(text):
    return re.sub('\n', ' ', text)


class TweetListener(StreamListener):
    # A listener handles tweets are the received from the stream.
    # This is a basic listener that just prints received tweets to standard output
    def __init__(self, tqueue=None, api=None):
        super(TweetListener, self).__init__()
        self.tweet_queue = tqueue
        self.num_tweets = 0
        self._file = open('Tweets.csv', 'w+', encoding='utf8')
        self._writer = csv.writer(self._file)
        self.tweets_list = []
        self._flag = True
        self._writer.writerow(
            ['author', 'date', 'text', 'hashtags', 'mentions', 'urls', 'truncated'])

    def on_status(self, status):
        try:

            # self.save_tweets(status)
            self.tweet_queue.append(status)
        except KeyboardInterrupt:
            sys.exit()
            return False

        return True

    def save_tweets(self, status):
        if not status.retweeted and 'RT @' not in status.text:
            if status.truncated:

                self._writer.writerow(
                    [status.author.screen_name,
                     status.created_at,
                     clean_text(status.__dict__
                                ['_json']['extended_tweet']['full_text']),
                     status.__dict__
                     ['_json']['extended_tweet']['entities']['hashtags'],
                     status.__dict__
                     ['_json']['extended_tweet']['entities']['user_mentions'],
                     status.__dict__
                     ['_json']['extended_tweet']['entities']['urls'],
                     status.truncated]
                )
            else:
                self._writer.writerow(
                    [status.author.screen_name,
                     status.created_at,
                     clean_text(status.text),
                     status.entities['hashtags'],
                     status.entities['user_mentions'],
                     status.entities['urls'],
                     status.truncated]
                )

            # json.dump(status._json, self._file)
            # print textwrap.fill(data.text, width=60, initial_indent='    ', subsequent_indent='    ')
            # print '\n %s  %s  via %s\n' % (data.author.screen_name, data.created_at, data.source)
            self.num_tweets += 1
            stdout.write(" Tweets count total: %s      %s" %
                         (self.num_tweets, "\r"))
        stdout.flush()

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_error disconnects the stream
            return False
        elif status_code == 406:
            # passing when invalid format
            pass
        print(sys.stderr, 'Encountered error with status code:', status_code)
        return True  # Don't kill the stream

    def on_timeout(self):
        print(sys.stderr, 'Timeout...')
        return True  # Don't kill the stream


def stream_tweets(tweets_queue):

    # The consumer key and secret will be generated for you after
    CONSUMER_KEY = 'ssPIaT3IfbO8WuQmvx9JxLLZk'
    CONSUMER_SECRET = 'iQfaRlzAliPzLOP28H0FkkfuucEnrLDhW0QjKBqSGxfEk2j4Dk'

    # Create an access token under the the "Your access token" section
    ACCESS_TOKEN = '1059687193730519040-7kATiofR37lbskR3HtvUxTQSjTi9ZW'
    ACCESS_TOKEN_SECRET = '15pbd8SBmO24PyqRq3R8IVnqemttFdRJg2fE5Cq84DZbb'

    # printing all the tweets to the standard output
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    stream = Stream(auth, TweetListener(tqueue=tweets_queue))
    try:
        stream.filter(track=['#coronavirus', '#corona', '#COVIDー19', '#covidiot', '#covidiots'],
                      languages=['en'], is_async=True)

    except KeyboardInterrupt:
        sys.exit()
        return False

    except ChunkedEncodingError:
        # Sometimes the API sends back one byte less than expected which results in an exception in the
        # current version of the requests library
        stream_tweets(tweet_queue)


def process_tweets(tweets_queue):
    global NUMTWEETS
    while True:
        if len(tweets_queue) > 0:

            cleaner = CleanProcess(tweets_queue[0])
            roundOne = cleaner.round1(tweets_queue[0])
            csv_processor.writerow(
                [
                    roundOne[0],
                    SentimentObject.sentiment(roundOne[0]),
                    roundOne[4]
                ])
            # print(len(tweets_queue[0]))
            tweets_queue.popleft()
            NUMTWEETS += 1
            stdout.write(" Tweets count total: %s      %s" %
                         (NUMTWEETS, "\r"))
        stdout.flush()
        #     #  Do something with the tweets
        #     print()


if __name__ == '__main__':
    NUMTWEETS = 0
    SentimentObject = Sentiment()
    processor = open('StreamingProcess.csv', 'w', encoding='utf8')
    csv_processor = csv.writer(processor)
    tweet_queue = deque()

    tweet_stream = Thread(target=stream_tweets,
                          args=(tweet_queue,), daemon=True)
    tweet_stream.start()
    process_tweets(tweet_queue)


# Load all classifiers from the pickled files
# function to load models given filepath
def load_model(file_path):
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier
