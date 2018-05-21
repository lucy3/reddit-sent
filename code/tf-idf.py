"""
Manual tf-idf since we only have word counts. 

The misalignment paper takes in consideration
"common" one to four grams but 
it's so time intensive to get ngrams.  
"""
import os
import math
from collections import Counter
import numpy as np
from scipy import spatial
from tqdm import tqdm # progress bar

UNIGRAMS = '../data/unigrams/'
OUTPUT = '../logs/tf-idf_unigrams'

def get_idf(): 
    '''
    For each word, get log_10(N/df) where N is 
    number of subreddits. 
    Set min_df to be 5 and max_df to be 0.95
    to filter out too rare and too common words.
    '''
    dfs = Counter() 
    num_sr = 0
    for sr in tqdm(os.listdir(UNIGRAMS)): 
        words = set() 
        if sr.startswith('.'): continue
        num_sr += 1
        for month in os.listdir(UNIGRAMS + sr): 
            if month.startswith('.'): continue
            with open(UNIGRAMS + sr + '/' + month, 'r') as inputfile: 
                for line in inputfile: 
                    words.add(line.split()[0])
        for w in words: 
            dfs[w] += 1
    idfs = {}
    for w in dfs: 
        if dfs[w] >= 5 and dfs[w] <= 0.95*num_sr: 
            idfs[w] = math.log10(num_sr/float(dfs[w]))
    return idfs
    

def get_tf_idf(idfs): 
    '''
    For each word in idfs, get its (1 + log tf) 
    for each document. 
    Create vectors for each document where each 
    index corresponds to a word and the value is
    (1 + log tf)xlog_10(N/df). 
    '''
    vocab = sorted(idfs.keys())
    srs = []
    X = []
    for sr in tqdm(os.listdir(UNIGRAMS)): 
        srs.append(sr)
        tfs = Counter() 
        if sr.startswith('.'): continue
        num_sr += 1
        for month in os.listdir(UNIGRAMS + sr): 
            if month.startswith('.'): continue
            with open(UNIGRAMS + sr + '/' + month, 'r') as inputfile: 
                for line in inputfile: 
                    contents = line.strip().split()
                    w = contents[0] 
                    if w in idfs: 
                        count = int(counters[-1])
                        tfs[w] += count
        vec = np.zeros(len(vocab))
        for i, w in enumerate(vocab): 
            if tfs[w] > 0: 
                vec[i] = (1 + math.log10(tfs[w]))*idfs[w]
        X.append(vec)
    X = np.array(X)
    print X.shape
    np.save(OUTPUT, X)
    print 1 - spatial.distance.cosine(X[srs.index('android')], X[srs.index('apple')])
    print 1 - spatial.distance.cosine(X[srs.index('london')], X[srs.index('ukpolitics')])
    print 1 - spatial.distance.cosine(X[srs.index('london')], X[srs.index('android')])

def main(): 
    idfs = get_idf() 
    get_tf_idf(idfs)

if __name__ == '__main__':
    main()