"""
Generates jsons from subreddit : # of tokens
and subreddit : # of comments for one year
of data. 
"""

from pyspark import SparkConf, SparkContext
import json
from collections import Counter
import time

SUBREDDITS = '../data/subreddits_no_defaults.txt'
MINI = '../data/mini_data.txt' # for code testing purposes
INPUT_PREFIX = '/dfs/dataset/infolab/Reddit/comments/'
COM_COUNTS = '../logs/comment_counts.json'
TOK_COUNTS = '../logs/token_counts.json'

def get_subreddit_tokens(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment: 
        length = len(comment['body'].split())
        return (comment['subreddit'].lower(), length)
    else: 
        return (None, 0)
    
def get_subreddit_comments(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment: 
        return (comment['subreddit'].lower(), 1)
    else: 
        return (None, 0)
    
def count_comments(): 
    start = time.time()
    reddits = set()
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            reddits.add(line.strip()[3:].lower())
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    all_token_counts = Counter()
    all_comment_counts = Counter()
    months = ['2016/RC_2016-05', '2016/RC_2016-06', '2016/RC_2016-07', \
              '2016/RC_2016-08', '2016/RC_2016-09', '2016/RC_2016-10', \
              '2016/RC_2016-11', '2016/RC_2016-12', '2017/RC_2017-01', \
              '2017/RC_2017-02', '2017/RC_2017-03', '2017/RC_2017-04']
    for m in months: 
        path = INPUT_PREFIX + m
        print path
        data = sc.textFile(path)
        token_counts = data.map(get_subreddit_tokens)
        comment_counts = data.map(get_subreddit_comments)
        token_counts = token_counts.filter(lambda l: l[0] in reddits)
        comment_counts = comment_counts.filter(lambda l: l[0] in reddits)
        token_counts = token_counts.reduceByKey(lambda n1, n2: n1 + n2)
        comment_counts = comment_counts.reduceByKey(lambda n1, n2: n1 + n2)
        token_counts_dict = token_counts.collectAsMap()
        comment_counts_dict = comment_counts.collectAsMap()
        all_token_counts.update(token_counts_dict)
        all_comment_counts.update(comment_counts_dict)
    with open(TOK_COUNTS, 'w') as outputfile: 
        json.dump(all_token_counts, outputfile)
    with open(COM_COUNTS, 'w') as outputfile: 
        json.dump(all_comment_counts, outputfile)
    sc.stop() 
    end = time.time()
    print "TIME:", end - start

def main(): 
    count_comments()

if __name__ == '__main__':
    main()