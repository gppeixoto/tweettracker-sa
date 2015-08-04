"""
# @author: Guilherme Peixoto  -- gpp@cin.ufpe.br    			#
#                                                   			#
# Script for parsing tweets from mongo-db's backup. It is 		#
# requeired to input through stdin. Usage example for files on  #
# the cluster is as follows:									#
#																#

  zcat path/to/tweettracker-tweets-2015-06-11.json.gz.Z | python parser.py lang-code limit

# where lang-code is the integer that maps to the language on db#
# e.g.: 9="en" and limit is the upper bound on the #tweets to   #
# be collected. 												#
# TODO: parse time span of tweets as args. Currently collected  #
# within Jan1st of year before execution. 		 				#
"""
import ujson as json
import sys
import time
from datetime import datetime as dt
from pprint import pprint as pretty
from numpy.random import rand as rnd

emojiList = set([':-)', '(-:', '=)', '(=', '(:', ':)', ':-(', ')-:', '=(', ')=', ':(', '):', ':D', '^_^', \
	'^__^', '^___^', ':d', 'd:', ': )', '( :', ': (', ') :', '8)', '(8', '8(', ')8', '8 )', ') 8', '8 (', ';)', \
	'(;', '; )', '( ;', ';-)', '(-;'])

t0 = time.time()
docs = []
p = 0.05
_limit = int(sys.argv[-1])
lang = sys.argv[-2]
curr_year = time.gmtime()[0]

for lineCount, line in enumerate(sys.stdin):
	if lineCount % 100000 == 0: 
		print '%d: %.0fs %d' % (lineCount, (time.time()-t0), len(docs))
	if limit != -1 and len(docs) > _limit-1:
		print '%d tweets collected at line %d' % (len(docs), lineCount)
		break
	doc = json.loads(line)
	if doc.has_key("tweet-lang") and doc["tweet-lang"] == lang:
		tweet = doc["text"]
		words = set(tweet.split()) # this could be improved using twokenize. However, it will
		# make the script severely slower and for collecting purposes I do not think it is necessary.
		if doc.has_key("timestamp") and words.intersection(emojiList) != set():
			timestamp = doc["timestamp"]["$numberLong"][:-3]
			timestamp = int(timestamp)
			year = dt.fromtimestamp(timestamp).year
			if abs(year - curr_year) <= 1:
				docs.append(doc["text"])

import cPickle as pickle
tt = time.gmtime()[:-3]
tt = "-".join([str(i) for i in tt])
print 'Saving...'
fname = "tweets-lang="+lang+"_"+tt+".p"
pickle.dump(docs, open(fname, "wb"))
print 'Tweets stored at: %s' % fname
print 'Time elapsed: %.0fs' % ((time.time()-t0))
