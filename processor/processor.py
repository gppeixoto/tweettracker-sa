"""
# @author: Guilherme Peixoto  -- gpp@cin.ufpe.br    #
#                                                   #
# Class for processing input to the adequate format #
#   to sentiment classification. Text is represent- #
#  -ed as a tf-idf matrix concatenated with a set   #
#   of hand-crated features. This representation is #
#   close to the state of the art for twitter.      #
#                                                   #
#   See:                                            #
#   Mohammad et. al, "NRC-Canada: Building the Sta- #
#   of-the-art in Sentiment Analysis of Tweets"     #
#   SemEval 2013.                                   #
# http://saifmohammad.com/WebDocs/sentimentMKZ.pdf  #
"""

from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.preprocessing import scale
import numpy as np
import re
import cPickle as pickle
from nltk.corpus import stopwords
import twokenize # tweet tokenizer
import time
from scipy.sparse import csr_matrix as toSparse
from scipy.sparse import hstack
import goslate
from warnings import warn
from sys import stdout

tweet_tokenizer = twokenize.tokenize
stop = stopwords.words('english')
# Taking important words off of the stopwords dictionary
stop.remove('not')
adverbs = set(['very', 'extremely', 'highly'])

for adv in adverbs: 
    if adv in stop: stop.remove(adv)

emojiList = set([':-)', '(-:', '=)', '(=', '(:', ':)', ':-(', ')-:', '=(', ')=', ':(', '):', ':D', '^_^', '^__^', '^___^', ':d', 'd:', \
    ': )', '( :', ': (', ') :', '8)', '(8', '8(', ')8', '8 )', ') 8', '8 (', ';)', '(;', '; )', '( ;', ';-)', '(-;'])

posEmoji = set([':-)', '(-:', '=)', '(=', '(:', ':)', ':-(', ':D', '^_^', '^__^', '^___^', ':d', 'd:', ': )', '( :', '8)', \
            '(8', '8 )', ';)', '; )', '; )', '( ;', ';-)', '(-;', '(;'])

negEmoji = emojiList.difference(posEmoji)

punctuation = set([',', ';', '.', ':', '.', '!', '?', '\"', '*', '\'', '(', ')', '-'])
pattern = re.compile(r'(.)\1{2,}', re.DOTALL) # for elongated words truncation

class Processor:
    """
    Class for processing tweets to the adequate format. Do not use for
    many languages at once, as it will not only cause a high drop in
    accuracy but may break the code as well.
    
    Parameters
    ----------
    tokenizer : instance of tokenizer class
        Tokenizer to be used on tokenizing tweets. If None,
        tweets will be split according to white space, which
        generates a lot of noisy data. Uses CMU TweetNLP tokenizer
        as default (http://www.ark.cs.cmu.edu/TweetNLP/)
    stopwords : set
        Set of words to be removed from tweets. Default collection
        is a set of stopwords for english.
    lang : string
        Language of the tweets. Default set to english.
    """
    def __init__(self, tokenizer=tweet_tokenizer, lang="en", stopwords=stop, ngrams=2):
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.lang = lang
        self.__target_not = "not"
            #TODO: use the adverbs below to generate lexicons in target language
        self.__adverbs = adverbs
        self.__translator = goslate.Goslate()
        # WARNING: do NOT change the parameters of the vectorization. It is already
        # set to the optimal configuration.
        self.__vectorizer = Tfidf(ngram_range=(1,ngrams), binary=True,
            tokenizer=self.tokenizer)
        self.__fitted = False

        if lang != "en":
            self.__target_not = self.__translator.translate("not", lang)
            self.__adverbs = [self.__translator.translate(adv, lang) for adv in adverbs]

    def get_params(self):
        params = dict()
        params["tokenizer"] = self.tokenizer
        params["lang"] = self.lang
        params["stopwords"] = list(self.stopwords) if self.stopwords != None else None
        #TODO: fix the other parameters
        return params

    def set_params(self, tokenizer=None, lang=None, stopwords=None, ngrams=None):
        #TODO: fix the other parameters
        if tokenizer is not None:
            setattr(self, "tokenizer", tokenizer)
        if lang is not None:
            setattr(self, "lang", lang)
        if stopwords is not None:
            setattr(self, "stopwords", stopwords)
        if ngrams is not None:
            setattr(self, "ngrams", ngrams)

    def __preprocess(self, tweetList, verbose=False):
        """
        Internal method for preprocessing the tweets. Do not call this
        directly, use "process" instead.
        """

        rep_count = []
        hst_count = []
        hst_last = []
        exc_count = []
        exc_last = []
        qst_count = []
        qst_last = []
        neg_count = []
        tw_length = [] 
        labels = [] 
        ll = len(tweetList)
        dot = ll / 50
        for x in xrange(ll):
            if dot > 0 and x % dot == 0:
                stdout.write("."); stdout.flush()
            tweet = tweetList[x].lower().encode('utf-8').decode('utf-8')

            # Count reps
            reps = pattern.findall(tweet)
            if reps != []: tweet = pattern.sub(r'\1\1', tweet)
            rep_count.append(len(reps))

            # Tokenizing
            tweet = self.tokenizer(tweet) # ok to use independent of language

            # Removing stopwords and retweet noise
            tweet = [word for word in tweet if word not in self.stopwords and not word.startswith('RT')]

            # Normalizing mentions, hyperlinks
            reps = 0. # float is intended type
            hsts = 0. # necessary for scaling
            excs = 0.
            qsts = 0.
            negs = 0.
            last = -1.
            label = np.inf
            for i, word in enumerate(tweet):
                if word.startswith(('.@', '@')): #mention
                    tweet[i] = '___mention___'
                if word.startswith(('www','http')):
                    tweet[i] = '___url___'
                if word.startswith('!'):
                    excs += 1
                    last = 0
                if word.startswith('?'): #TODO: problem with ?!, !?, account for this
                    qsts += 1
                    last = 1
                if word.startswith('#'):
                    hsts += 1
                    last = 2
                if word == self.__target_not:
                    negs += 1
                    tweet[i] = ''
                    if (i+1)<len(tweet):
                        tweet[i+1] = self.__target_not+'___'+tweet[i+1]
                    else:
                        tweet[i] = self.__target_not
                if label == np.inf and word in posEmoji:
                    label = +1
                elif label == np.inf and word in negEmoji:
                    label = -1
            hst_count.append(hsts)
            qst_count.append(qsts)
            exc_count.append(excs)
            neg_count.append(negs)
            tw_length.append(len(tweet))
            labels.append(label)
            # Removing punctuation
            tweet = [''.join([w for w in word if w not in punctuation]) for word in tweet if len(word)>2]
            tweet = ' '.join(tweet) 
            tweetList[x] = tweet
        return (tweetList, rep_count, hst_count, exc_count, qst_count, neg_count, tw_length, labels)

    def process(self, tweetList, verbose=False):
        """
        Processes the tweet list and returns the corpus after the
        normalization process and the features matrix.

        Parameters
        ----------
        tweetList : list
            List of tweets to be processed.

        Returns
        ----------
        corpus : list
            List of tweets after the normalization process.
        feats : (N,5) numpy matrix
            Dense matrix of Twitter features.
            feats[0]: number of elongated words.
            feats[1]: number of hashtags.
            feats[2]: number of exclamation points.
            feats[3]: number of question marks.
            feats[4]: number of negated contexts.
            feats[5]: number of words.
            @TODO: include the clusters counts.
        """
        t0 = time.time()
        if verbose:
            print 'Normalizing and extracting features'
        ret = self.__preprocess(tweetList, verbose)
        corpus = ret[0]
        rep_count, hst_count, exc_count, qst_count, neg_count, \
            tw_length, labels = map(lambda x: np.array(x), list(ret[1:]))
        feats = np.vstack((rep_count, hst_count, exc_count, qst_count, \
                            neg_count, tw_length, labels)).transpose()
        if verbose:
            print '\nTime elapsed on processing and feature extraction: %.0fs' % ((time.time()-t0))
        return (corpus, feats)

    def fit_vectorizer(self, tweetList, saveVectorizer=False, verbose=False):
        """
        Fits the tweet list to the instance's vectorizer.

        Parameters
        ----------
        tweetList : list
            Normalized tweet list, each entry containing one tweet only.
        saveVectorizer : boolean, optional
            Whether to save the vectorizer in disk (binary format). 
            Default set to false.
        verbose : boolean , optional
            Set verbose output on or off.        
        """
        t0 = time.time()
        if verbose: print 'Fitting tweets to vectorizer...'
        self.__vectorizer.fit(tweetList)
        if not self.__fitted:
            self.__fitted = True
        if verbose:
            print 'Time elapsed on fitting: %.0fs' % ((time.time()-t0))

        if saveVectorizer == True:
            if verbose: print 'Storing vectorizer in disk...'
            t0 = time.time()
            tt = time.localtime()[:-3]
            tt = '-'.join([str(i) for i in list(tt)])
            pickle.dump(self.__vectorizer, open("vectorizer-"+self.lang+"_"+tt+".p", "wb"))
            if verbose:
                print 'Time elapsed storing vectorizer: %.0fs' % ((time.time()-t0))

    def transform(self, tweetList, saveMatrix=False, verbose=False):
        """
        Transform each tweet on the tweet list according to the
        fitted vectorizer.

        Parameters
        ----------
        tweetList : list
            Normalized tweet list, each entry containing one tweet only.
        saveMatrix : boolean, optional
            Whether to save the matrix in disk (binary format). 
            Default set to false.
        verbose : boolean , optional
            Set verbose output on or off.

        Returns
        ----------
        mat : csr_matrix
            Sparse matrix representing the TF-IDF frequency
            counts on the tweets.           
        """
        if not self.__fitted:
            warn("Calling transform before the vectorizer was ever fitted")
        if verbose: print 'Vectorizing the tweets...'
        t0 = time.time()
        mat = self.__vectorizer.transform(tweetList)
        if verbose:
            print 'Time elapsed vectorizing %d samples: %.0fs' % (len(tweetList), (time.time()-t0))

        if saveMatrix == True:
            if verbose: print 'Storing the matrix in disk...'
            t0 = time.time()
            tt = time.localtime()[:-3]
            tt = '-'.join([str(i) for i in list(tt)])
            pickle.dump(self.__vectorizer, open("matrix-"+self.lang+"_"+tt+".p", "wb"))
            if verbose:
                print 'Time elapsed storing %d samples: %.0fs' % (len(tweetList), (time.time()-t0))

        return mat

    def fit_transform(self, tweetList, saveVectorizer=False, saveMatrix=False, verbose=False):
        """
        Convenience method. Calls \"fit_vectorizer\" followed by
        \"transform\".
        
        Parameters
        ----------
        tweetList : list
            Normalized tweet list, each entry containing one tweet only.
        saveVectorizer : boolean, optional
            Whether to save the vectorizer in disk (binary format). 
            Default set to false.
        saveMatrix : boolean, optional
            Whether to save the matrix in disk (binary format). 
            Default set to false.
        verbose : boolean , optional
            Set verbose output on or off.

        Returns
        ----------
        mat : csr_matrix
            Sparse matrix representing the TF-IDF frequency
            counts on the tweets.           
        """
        self.fit_vectorizer(tweetList, saveVectorizer, verbose)
        mat = self.transform(tweetList, saveMatrix, verbose)
        return mat

    def build_feature_matrix(self, tweetList, useTwitterFeatures, fit, saveVectorizer=False, 
            saveMatrix=False, verbose=False):
        """
        Convenience method. Fits the vectorizer to \"raw\" tweet list input 
        and builds the feature matrix. If vectorizer has already been fitted, 
        invokes \"transform\" instead of \"fit_transform\". Twitter-specific 
        features can be either concatenated or not.

        Parameters
        ----------
        tweetList : list
            Normalized tweet list, each entry containing one tweet only.
        useTwitterFeatures : boolean
            Whether to concatenate Twitter-specific features to the feature
            matrix. The features are determined as in \"process\" function.
        fit : boolean
            Whether the vectorizer should be fitted or not. If True, function
            will invoke \"fit_transform\", otherwise \"transform\".
        saveVectorizer : boolean, optional
            Whether to store the vectorizer in disk (binary format). 
            Default set to false.
        saveMatrix : boolean, optional
            Whether to store the resulting matrix in disk (binary format). 
            Default set to false.
        verbose : boolean , optional
            Set verbose output on or off.

        Returns
        ----------
        mat : csr_matrix
            Feature matrix for the tweet list.
        labels : array-like
            Label array for the tweet list.        
        """
        corpus, feats = self.process(tweetList, verbose)
        labels = feats[:, -1]
        feats = feats[:, :-1]
        if fit:
            mat = self.fit_transform(corpus, saveVectorizer, saveMatrix, verbose)
        else:
            mat = self.transform(corpus, saveMatrix, verbose)
        if useTwitterFeatures:
            feats = scale(feats) # scale needed to faster convergence
            feats = toSparse(feats) # csr_matrix format
            mat = hstack([mat, feats])
        return (mat, labels)

