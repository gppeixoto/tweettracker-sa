"""
# @author: Guilherme Peixoto  -- gpp@cin.ufpe.br    #
#                                                   #
# Simple use case for the processor class. Data is  #
#   a subset of Sentiment140 dataset.               #
#   http://help.sentiment140.com/for-students       #
#   This script uses a .p format different from the #
#   raw data provided by Sentiment140. TODO: fix    #
#   the input format or provide the datas.          #
#   XYZ min to run on a i5 2.7GHz.                  # 
"""
import cPickle as pickle
from sklearn.linear_model import LogisticRegression as LR
from Processor import Processor
import numpy as np

X_train = pickle.load(open("labeledTrainingData.p","r"))
X_train, y_train = zip(*X_train)
X_train, y_train = map(lambda x: list(x), [X_train, y_train])

X_test = pickle.load(open("test_data.p", "r"))
X_test, y_test = zip(*X_test)
X_test, y_test = map(lambda x: list(x), [X_test, y_test])

pr = Processor()
# ~11min on line below
X_train_mat, idxs = pr.process_build(X_train, verbose=True)
# Since removeShort=True (default), then we capture the indexes returned as well so we don't lose
#   track of the labels association to the samples
y_train = np.array(y_train)[idxs]

clf = LR()
clf.fit(X_train_mat, y_train)

preds = clf.predict(pr.transform(X_test))
# it should yield 83%~
np.sum([int(x==y) for x,y in zip(preds, y_test)])/(len(preds)+.0)