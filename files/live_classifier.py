import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class EnsembleClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


class Sentiment():
    def __init__(self):

        # Original Naive Bayes Classifier
        self.ONB_Clf = self.load_model('pickled_algos/ONB_clf.pickle')

        # Multinomial Naive Bayes Classifier
        self.MNB_Clf = self.load_model('pickled_algos/MNB_clf.pickle')

        # Bernoulli  Naive Bayes Classifier
        self.BNB_Clf = self.load_model('pickled_algos/BNB_clf.pickle')

        # Logistic Regression Classifier
        self.LogReg_Clf = self.load_model('pickled_algos/LogReg_clf.pickle')

        # Stochastic Gradient Descent Classifier
        self.SGD_Clf = self.load_model('pickled_algos/SGD_clf.pickle')

        self.ensemble_clf = EnsembleClassifier(
            self.ONB_Clf, self.MNB_Clf, self.BNB_Clf, self.LogReg_Clf, self.SGD_Clf)

    def parse_text(self, document):
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    # Load all classifiers from the pickled files
    # function to load models given filepath

    def load_model(self, file_path):
        classifier_f = open(file_path, "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        return classifier

    def sentiment(self, text):
        feats = self.parse_text(text)
        return self.ensemble_clf.classify(feats), self.ensemble_clf.confidence(feats)


# SentimentObject = Sentiment()
# text_a = '''The problem is with the corporate anticulture that controls these productions-and
#             the fandom-targeted demagogy that they're made to fulfill-which responsible casting
#                 can't overcome alone.'''
# print(SentimentObject.sentiment(text_a))
