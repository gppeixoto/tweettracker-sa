# tweettracker-sa
A sentiment analysis tool for Twitter.

Check [twitter-specific tokenizing repository](https://github.com/myleott/ark-twokenize-py) for updates and bug fixing. 

### Required Dependencies
    - Scikit-Learn
    - NumPy
    - NLTK
    - SciPy
    - Goslate

### References
1. Go, Alec, Richa Bhayani, and Lei Huang. ["Twitter sentiment classification using distant supervision."](http://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf) CS224N Project Report, Stanford 1 (2009): 12.
2. Mohammad, Saif M., Svetlana Kiritchenko, and Xiaodan Zhu. ["NRC-Canada: Building the state-of-the-art in sentiment analysis of tweets."](http://www.aclweb.org/website/old_anthology/S/S13/S13-2.pdf#page=357) Second Joint Conference on Lexical and Computational Semantics (* SEM). Vol. 2. 2013.
3. Owoputi, Olutobi, et al. ["Improved part-of-speech tagging for online conversational text with word clusters."](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1039&context=lti) Association for Computational Linguistics, 2013.

### Dataset
1. Dataset for `distant_supervision.py` example provided by [Sentiment140](http://help.sentiment140.com/for-students/). You can directly download the corpus [here](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip).
2. Dataset for `demonstration.py` example from TweetTracker's backup.

### Usage
- Use `Processor` class for tweet processing and vectorizing
- Use `parse.py` for collecting proper tweets from TweetTracker's backup to fit the classifier.

### How to:
Before deployment, the following steps should be done:
1. Collect data using `parse.py` from tweet:  
    `zcat path/to/backup_file.json.gz.Z | python parse.py lang limit`  
where **lang** stands for language (either code or abbreviation, depending on the "tweet-lang" type) and limit is the number of tweets to be collected (pass -1 to collect every possible tweet).  
    - `WARNING (1):` there should be one classifier per language, therefore each either a bash script is needed to make this automated for many languages or call for each desired language.
2. Process the data collected with `Processor` class. The following settings are default:  
    + TF-IDF representation of the vocabulary
    + Unigrams and bigrams only (trade-off between processing time complexity and accuracy improvement by using trigrams favors time not using trigrams)
    + Twitter-specific features are **not concatened** by default, must set the parameter on (see documentation for usage)
    - `WARNING (2):` the tokenizer makes mistakes and sometimes the label from the emoticons is not inferred, therefore it can't be used for classification. Since those instances are few, we can simply discard these samples. Use `Processor.clear`method for this.
3. Store the vectorizer fitted from the data for each language. This will be necessary to classify online, unseen data.
4. Use `sklearn.linear_model.LogisticRegression` as classifier for optimal results. Using a LinearSVM does not yield a big accuracy improvement, is not as fast and is not possible to easily get the probabilities.





