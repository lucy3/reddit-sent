"""
Generates jsons from subreddit : # of tokens
and subreddit : # of comments for one year
of data. 
"""

from pyspark import SparkConf, SparkContext
import json
from collections import Counter
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SUBREDDITS = '/dfs/scratch2/lucy3/reddit-sent/data/subreddits_no_defaults.txt'
OUR_SR = '/dfs/scratch2/lucy3/reddit-sent/data/our_subreddits.txt'
MINI = '/dfs/scratch2/lucy3/reddit-sent/data/mini_data.txt' # for code testing purposes
INPUT_PREFIX = '/dfs/dataset/infolab/Reddit/comments/'
COM_COUNTS = '/dfs/scratch2/lucy3/reddit-sent/logs/comment_counts.json'
TOK_COUNTS = '/dfs/scratch2/lucy3/reddit-sent/logs/token_counts.json'
COM_HIST = '/dfs/scratch2/lucy3/reddit-sent/logs/comment_hist.png'
TOK_HIST = '/dfs/scratch2/lucy3/reddit-sent/logs/token_hist.png'

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
    
def get_hists(): 
    reddits = set()
    with open(OUR_SR, 'r') as inputfile: 
        for line in inputfile: 
            reddits.add(line.strip().lower())
    with open(COM_COUNTS, 'r') as inputfile: 
        data = json.load(inputfile)
    vals = []
    for sr in data: 
        if sr.lower() in reddits: 
            vals.append(data[sr])
    print max(vals), min(vals)
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=30)
    plt.yscale('log')
    plt.ylabel('log frequency')
    plt.xlabel('# of comments')
    plt.title('Subreddit Comment Counts')
    plt.savefig(COM_HIST)
    plt.close()
    with open(TOK_COUNTS, 'r') as inputfile: 
        data = json.load(inputfile)
    vals = []
    for sr in data: 
        if sr.lower() in reddits: 
            vals.append(data[sr])
    print max(vals), min(vals)
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=30)
    plt.yscale('log')
    plt.ylabel('log frequency')
    plt.xlabel('# of tokens')
    plt.title('Subreddit Token Counts')
    plt.savefig(TOK_HIST)
    plt.close()

def main(): 
    get_hists()

if __name__ == '__main__':
    main()