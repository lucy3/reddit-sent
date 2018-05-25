"""
UGH 

Iterate through data and create giant
subreddit documents
"""
import json
import time
from tqdm import tqdm
import re
import string
import os
from pyspark import SparkConf, SparkContext

SUBREDDITS = '../data/our_subreddits.txt'
INPUT_PREFIX = '/dfs/dataset/infolab/Reddit/comments/'
MINI = '../data/mini_data.txt' # for code testing purposes
OUTPUT = '../data/partial_docs/'
regex = re.compile('[%s]' % re.escape(string.punctuation))
reddits = set()
            
def get_comment(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment and \
        comment['subreddit'].lower() in reddits: 
        text = comment['body'].lower()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = regex.sub('', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        return (comment['subreddit'].lower(), text)
    else: 
        return (None, '')
    
def save_doc(item): 
    if item[0] is not None:
        path = OUTPUT + item[0] + '/'
        with open(path + MONTH.split('/')[-1], 'w') as file:
            file.write(item[1].encode('utf-8', 'replace'))
        
def parallel_ugh(): 
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            reddits.add(line.strip().lower())
    for sr in reddits: 
        path = OUTPUT + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)
    months = ['2016/RC_2016-05', '2016/RC_2016-06', '2016/RC_2016-07', \
              '2016/RC_2016-08', '2016/RC_2016-09', '2016/RC_2016-10', \
              '2016/RC_2016-11', '2016/RC_2016-12', '2017/RC_2017-01', \
              '2017/RC_2017-02', '2017/RC_2017-03', '2017/RC_2017-04']
    conf = SparkConf()
    for m in months: 
        sc = SparkContext(conf=conf)
        global MONTH
        MONTH = m
        start = time.time()
        path = INPUT_PREFIX + m
        print path
        data = sc.textFile(path)
        data = data.map(get_comment)
        data = data.reduceByKey(lambda n1, n2: n1 + ' \n\n ' + n2)
        data = data.foreach(save_doc)
        print "TIME", time.time() - start
        sc.stop()

def main(): 
    parallel_ugh()

if __name__ == '__main__':
    main()