import cPickle as pickle
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score
from processor import Processor
import numpy as np
import time
import sys

fpath = sys.argv[-1]
tweets = pickle.load(open(fpath, "r"))
pr = Processor()
mat, labels = pr.build_feature_matrix(tweets, True, True, verbose=True)

def get_mask(labels):
    idxs = []
    for i, label in enumerate(labels):
        if label in set([-1, 1]): idxs.append(i)
    return np.array(idxs)

mask = get_mask(labels)
mat = mat[mask]
labels = labels[mask]

cut = int(.7*mat.shape[0])

X_train, y_train = mat[:cut], labels[:cut]
X_test, y_test = mat[cut:], labels[cut:]

clf = LR()
# Roughly 3 minutes on training
t0 = time.time()
print 'Training on %d samples...' % (X_train.shape[0])
clf.fit(X_train, y_train)
print 'Training time: %.0fs' % ((time.time()-t0))

print 'Testing on %d samples...' % (X_test.shape[0])
y_pred = clf.predict(X_test)

acc = (y_pred==y_test).sum()/(len(y_pred)+.0)
f1 = prfs(y_test, y_pred, average="macro")[-2]
roc_auc = roc_auc_score(y_test, y_pred)

print '\nReport\n'+'-'*40
print 'Accuracy: %.4f\nMacro F-1 Score: %.4f\nROC_AUC Score: %.4f' % (acc, f1, roc_auc)