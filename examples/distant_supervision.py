"""
# @author: Guilherme Peixoto  -- gpp@cin.ufpe.br    #
#                                                   #
# Simple use case for the processor class. Data is  #
#   a subset of Sentiment140 dataset.               #
#   http://help.sentiment140.com/for-students       #
#   This script uses a .p format different from the #
#   raw data provided by Sentiment140. TODO: fix    #
#   the input format or provide the datas.          #
#   Measurements run on a i5 2.7GHz.                # 
"""
import cPickle as pickle
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score
from Processor import Processor
import numpy as np
import time
import assert

# 30s on loading phase
t0 = time.time()
X_train = pickle.load(open("../data/experiment/labeledTrainingData.p","r"))
X_train, y_train = zip(*X_train)
X_train, y_train = map(lambda x: list(x), [X_train, y_train])

X_test = pickle.load(open("../data/experiment/test_data.p", "r"))
X_test, y_test = zip(*X_test)
X_test, y_test = map(lambda x: list(x), [X_test, y_test])
print '%.0fs' % ((time.time()-t0))

pr = Processor()

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

print 'Training on %d samples' % (X_mat.shape[0])
clf.fit(X_mat, y_train)
print 'Testing on %d samples' % (X_test.shape[0])
preds = clf.predict(X_test)

acc = np.sum([int(x==y) for x,y in zip(preds, y_test)])/(len(preds)+.0)
f1 = prfs(y_test, preds, average="macro")[-2]
roc_auc = roc_auc_score(y_test, preds)

print 'Accuracy: %.2f\t Macro F-1 Score: %.2f\t ROC_AUC Score: %.2f' % (acc, f1, roc_auc)