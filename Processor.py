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
from sklearn.preprocessing import scale as scaleMat
import numpy as np
import re
import cPickle as pickle
from nltk.corpus import stopwords
import twokenize # tweet tokenizer
import time
from scipy.sparse import csr_matrix as toSparse
from scipy.sparse import hstack
import goslate

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
    def __init__(self, tokenizer=tweet_tokenizer, stopwords=stop, lang="en", ngrams=2):
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

        if lang != "en":
            self.__target_not = self.__translator.translate("not", lang)
            self.__adverbs = [self.__translator.translate(adv, lang) for adv in adverbs]


    def __preprocess(self, tweetList):
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
                
        for x in xrange(len(tweetList)):
            tweet = tweetList[x].lower().decode('utf-8')

            # Count reps
            reps = pattern.findall(tweet)
            if reps != []: tweet = pattern.sub(r'\1\1', tweet)
            rep_count.append(len(reps))

            # Tokenizing
            tweet = self.tokenizer(tweet) # ok to use independent of language
            tw_length.append(len(tweet))

            # Short and stopwords, retweet and emoji removal
            tweet = [word for word in tweet if word not in emojiList\
                        and word not in self.stopwords and not word.startswith('RT')]

            # Normalizing mentions, hyperlinks
            reps = 0
            hsts = 0
            excs = 0
            qsts = 0
            negs = 0
            last = -1
            for i, word in enumerate(tweet):
                if word.startswith(('.@', '@')): #mention
                    tweet[i] = '___MENTION___'
                if word.startswith(('www','http')):
                    tweet[i] = '___URL___'
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
                    tweet[i+1] = 'NEG___' + tweet[i+1]
            hst_count.append(hsts)
            qst_count.append(qsts)
            exc_count.append(excs)
            neg_count.append(negs)

            # Removing punctuation
            tweet = [''.join([w for w in word if w not in punctuation]) for word in tweet if len(word)>2]
            tweet = ' '.join(tweet) 
            tweetList[x] = tweet
        return (tweetList, rep_count, hst_count, exc_count, qst_count, neg_count, tw_length)

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
            print 'Normalizing and extracting features...'
        ret = self.__preprocess(tweetList)
        corpus = ret[0]
        rep_count, hst_count, exc_count, qst_count, neg_count, \
            tw_length = map(lambda x: np.array(x), list(ret[1:]))
        feats = np.vstack((rep_count, hst_count, exc_count, qst_count, neg_count, tw_length)).transpose()
        if verbose:
            print 'Time elapsed on feature extraction: %.0fs' % ((time.time()-t0))
        return (corpus, feats)

    def build_matrix(self, tweetList, feats, removeShort=True, 
            saveVectorizer=False, saveMatrix=False, scale=True, verbose=False):
        """
        Parameters
        ----------
        tweetList : list
            Normalized tweet list, each entry containing one tweet only.
        feats : (N, 5) matrix
            Feature counts of each 
        removeShort : boolean
            Whether to remove tweets too short from the matrix.
            Short is considered less than 4 words. Default set 
            to true.
        saveVectorizer : boolean
            Whether to save the vectorizer in disk (binary format). 
            Default set to false.
        saveMatrix : boolean
            Whether to save (binary format)the sparse matrix 
            representation of the tweetList in disk.
        scale : boolean
            Whether to scale according to the variance.

        Returns
        ----------
        out : (N', M') matrix
            Sparse matrix with the Tfidf representation of unigrams,
            bigrams and trigrams plus the features counts. 
            @TODO: add the clusters counts.
        idxs : (N',) array, optional
            Returned in case removeShort=True. Array of indexes with the positions 
            of the tweets that were not removed in relation to the original dataset.
        """
        list_copy = []
        feat_copy = []
        idxs = []
        t00 = time.time()

        if removeShort:
            if verbose: print 'Removing short tweets...'
            for i, tweet in enumerate(tweetList):
                if feats[i][-1] > 3:
                    list_copy.append(tweet)
                    feat_copy.append(feats[i])
                    idxs.append(i)
        else:
            list_copy = tweetList
            feat_copy = feats
            idxs = None
        idxs = np.array(idxs)

        # Line below should take a while
        t0 = time.time()
        if verbose: print 'Vectorizing the tweets...'
        mat = self.__vectorizer.fit_transform(list_copy)
        if verbose:
            print 'Time elapsed on vectorizing process: %.0fs' % ((time.time()-t0))

        if saveVectorizer == True:
            if verbose: print 'Storing the vectorizer in disk...'
            t0 = time.time()
            pickle.dump(self.__vectorizer, open("vectorizer-"+self.lang+".p", "wb"))
            if verbose:
                print 'Time elapsed vectorizing: %.0fs' % ((time.time()-t0))
        if saveMatrix == True:
            if verbose: print 'Storing the vectorizer in disk...'
            t0 = time.time()
            if verbose:
                print 'Time elapsed storing matrix: %.0fs' % ((time.time()-t0))
            pickle.dump(mat, open("tweetMatrix-"+self.lang+".p", "wb"))

        mat = hstack([mat, toSparse(feat_copy)])
        if scale:
            mat = scaleMat(mat, with_mean=False)
        
        if idxs is None:
            return mat
        else:
            return (mat, idxs)

    def process_build(self, tweetList, verbose=False):
        """
        Convenience method. Processes input and builds matrix.

        Parameters
        ----------
        tweetList : list
            List of tweets to be processed.

        Returns
        ----------
        mat : (N', M) csr_matrix
            Sparse matrix representation of the extracted
            features from the tweets.
        idxs : (N',) array, optional
            Returned in case removeShort=True. Array of indexes with the positions 
            of the tweets that were not removed in relation to the original dataset.
        """
        
        corpus, feats = self.process(tweetList, verbose=verbose)
        mat = self.build_matrix(corpus, feats, saveVectorizer=False, saveMatrix=False, verbose=verbose)
        return mat

    def transform(self, tweetList):
        """
        Transform tweets to the trained tfidf representation. Do NOT
        call this function unless a vectorizer has already been trained.
        
        Parameters
        ----------
        tweetList : list
            List of tweets to be processed.

        Returns
        ----------
        mat : (N', M) csr_matrix
            Sparse matrix representation of the extracted
            features from the tweets.
        """
        tweets, feats = self.process(tweetList)
        mat = self.__vectorizer.transform(tweets)
        mat = hstack([mat, toSparse(feats)])
        mat = scaleMat(mat, with_mean=False)
        
        return mat