"""
# @author: Guilherme Peixoto  -- gpp@cin.ufpe.br    #
#                                                   #
# Simple use case for the processor class. Data is  #
#   a subset of Sentiment140 dataset.               #
#   For this script to run properly, you need       #
#   to have the decompressed zip file on the same   #
#   folder as this script, as well as the Processor #
#   class file.                                     #
#                                                   #
#   Should take close to 20min to run everything.   #
#   Measurements run on a 2.6 GHz Intel Core i5.    #
"""
import cPickle as pickle
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score
from Processor import Processor
import numpy as np
import time

def format(fpath):
    f = open(fpath)
    lines = [line.split("\",\"") for line in f]
    lines = [(line[-1][:-1], int(line[0][1:])) for line in lines]
    lines_return = []
    for i, line in enumerate(lines):
        label = -1
        if line[1] == 4:
            lines_return.append((line[0], +1))
        elif line[1] == 0:
            lines_return.append((line[0], -1))
        # ignoring neutral tweets and returning only
        #   those classified as either pos or neg for
        #   binary classification
    return lines_return

X_train = format("trainingandtestdata/training.1600000.processed.noemoticon.csv")
X_train, y_train = zip(*X_train)
X_train, y_train = map(lambda x: list(x), [X_train, y_train])

X_test = format("trainingandtestdata/testdata.manual.2009.06.14.csv")
X_test, y_test = zip(*X_test)
X_test, y_test = map(lambda x: list(x), [X_test, y_test])

pr = Processor()

print 'No twitter-specific features'
print '#'*40

# ~8min processing phase
# The twitter-specific tokenizer makes the parsing slow, however
#   the accuracy is much improved with it.
X_train, train_feats = pr.process(X_train, verbose=True)

# ~7min vectoring phase
X_mat = pr.fit_transform(X_train, saveVectorizer=False, saveMatrix=False, verbose=True)
X_test, test_feats = pr.process(X_test, verbose=True)
X_test = pr.transform(X_test, saveMatrix=False, verbose=True)

# Compare the accuracy with and w/o the twitter-specific features.
# Must scale the features matrix before concatenating with the ngrams matrix.
print 'TF-IDF Unigrams and Bigrams || Logistic Regression classifier'
print '-'*40

clf = LR()
# Roughly 3 minutes on training
t0 = time.time()
print 'Training on %d samples' % (X_mat.shape[0])
clf.fit(X_mat, y_train)
print 'Training time: %.0f' % ((time.time()-t0))

print 'Testing on %d samples' % (X_test.shape[0])
y_pred = clf.predict(X_test)

acc = (y_pred==y_test).sum()/(len(y_pred)+.0)
f1 = prfs(y_test, y_pred, average="macro")[-2]
roc_auc = roc_auc_score(y_test, y_pred)

print 'Report\n'+'-'*40
print 'Accuracy: %.4f\nMacro F-1 Score: %.4f\nROC_AUC Score: %.4f' % (acc, f1, roc_auc)