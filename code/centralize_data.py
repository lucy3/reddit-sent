"""
Takes the original input files and creates
for each subreddit three files per month
- 1) bigram counts of words
- 2) unigram counts of words
- 2) a list of username \t num_comments
"""
from pyspark import SparkConf, SparkContext
import json
from collections import Counter
import time
import os
import string
import re
from nltk import ngrams

SUBREDDITS = '/dfs/scratch2/lucy3/reddit-sent/data/our_subreddits.txt'
MINI = '/dfs/scratch2/lucy3/reddit-sent/data/mini_data.txt' # for code testing purposes
INPUT_PREFIX = '/dfs/dataset/infolab/Reddit/comments/'
regex = re.compile('[%s]' % re.escape(string.punctuation))

def get_subreddit_unigrams(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment: 
        text = comment['body'].lower()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = regex.sub('', text)
        tokens = text.split()
        return (comment['subreddit'].lower(), Counter(tokens))
    else: 
        return (None, Counter())
    
def get_subreddit_bigrams(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment: 
        text = comment['body'].lower()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = regex.sub('', text)
        tokens = text.split()
        counts = Counter()
        for i in range(len(tokens) - 1): 
            counts[tokens[i] + ' ' + tokens[i + 1]] += 1
        return (comment['subreddit'].lower(), counts)
    else: 
        return (None, Counter())
    
def get_subreddit_user(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'author' in comment: 
        return ((comment['subreddit'].lower(), comment['author']), 1)
    else: 
        return (None, 0)
    
def do_unigrams(data, m, reddits):
    uni = data.map(get_subreddit_unigrams)
    uni = uni.reduceByKey(lambda n1, n2: n1 + n2)
    uni = uni.filter(lambda l: l[0] in reddits)
    uni_dict = uni.collectAsMap()
    for sr in uni_dict: 
        path = '/dfs/scratch2/lucy3/reddit-sent/data/unigrams/' + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)
        file_path = path + m.split('/')[1]
        with open(file_path, 'w') as outputfile: 
            for tok in uni_dict[sr]: 
                outputfile.write(tok.encode('utf-8', 'replace') + '\t' + str(uni_dict[sr][tok]) + '\n')
                
def do_bigrams(data, m, reddits): 
    bi = data.map(get_subreddit_bigrams)
    bi = bi.reduceByKey(lambda n1, n2: n1 + n2)
    bi = bi.filter(lambda l: l[0] in reddits)
    bi_dict = bi.collectAsMap()
    for sr in bi_dict: 
        path = '/dfs/scratch2/lucy3/reddit-sent/data/bigrams/' + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)
        file_path = path + m.split('/')[1]
        with open(file_path, 'w') as outputfile: 
            for tok in bi_dict[sr]: 
                outputfile.write(tok.encode('utf-8', 'replace') + '\t' + str(bi_dict[sr][tok]) + '\n')
                
def do_users(data, m, reddits): 
    users = data.map(get_subreddit_user)
    users = users.reduceByKey(lambda n1, n2: n1 + n2) 
    users = users.map(lambda l: (l[0][0], set([(l[0][1], l[1])]))) 
    users = users.filter(lambda l: l[0] in reddits)
    users = users.reduceByKey(lambda n1, n2: n1 | n2) 
    users_dict = users.collectAsMap()
    for sr in users_dict: 
        path = '/dfs/scratch2/lucy3/reddit-sent/data/user/' + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)
        file_path = path + m.split('/')[1]
        with open(file_path, 'w') as outputfile: 
            for usr in users_dict[sr]: 
                outputfile.write(usr[0].encode('utf-8', 'replace') + '\t' + str(usr[1]) + '\n')

def filter_data(): 
    reddits = set()
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            reddits.add(line.strip().lower())
    months = ['2016/RC_2016-05', '2016/RC_2016-06', '2016/RC_2016-07', \
              '2016/RC_2016-08', '2016/RC_2016-09', '2016/RC_2016-10', \
              '2016/RC_2016-11', '2016/RC_2016-12', '2017/RC_2017-01', \
              '2017/RC_2017-02', '2017/RC_2017-03', '2017/RC_2017-04']
    conf = SparkConf()
    '''
    sc = SparkContext(conf=conf)
    for m in months: 
        start = time.time()
        path = INPUT_PREFIX + m
        print "BIGRAMS", path
        data = sc.textFile(path)
        do_bigrams(data, m, reddits)
        print "TIME:", time.time() - start
    sc.stop() 
    '''
    sc = SparkContext(conf=conf)
    for m in months: 
        start = time.time()
        path = INPUT_PREFIX + m
        print "USERS", path
        data = sc.textFile(path)
        do_users(data, m, reddits)
        print "TIME:", time.time() - start
    sc.stop() 
    sc = SparkContext(conf=conf)
    for m in months: 
        start = time.time()
        path = INPUT_PREFIX + m
        print "UNIGRAMS", path
        data = sc.textFile(path)
        do_unigrams(data, m, reddits)  
        print "TIME:", time.time() - start
    sc.stop() 

def main(): 
    filter_data()

if __name__ == '__main__':
    main()