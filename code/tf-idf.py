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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

UNIGRAMS = '/dfs/scratch2/lucy3/reddit-sent/data/unigrams/'
UNI_TFIDF = '/dfs/scratch2/lucy3/reddit-sent/logs/tf-idf_unigrams.npy'
UNI_SVD_TFIDF = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_unigrams.npy'
UNI_VOCAB = '/dfs/scratch2/lucy3/reddit-sent/logs/unigram_vocab'
UNI_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/unigram_rows'

USERS = '/dfs/scratch2/lucy3/reddit-sent/data/user/'
USR_TFIDF = '/dfs/scratch2/lucy3/reddit-sent/logs/tf-idf_users.npy'
USR_SVD_TFIDF = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_users.npy'
USR_VOCAB = '/dfs/scratch2/lucy3/reddit-sent/logs/users_vocab'
USR_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/users_rows'

def get_idf(rep): 
    '''
    For each word, get log_10(N/df) where N is 
    number of subreddits. 
    Set min_df to be 5 and max_df to be 0.95
    to filter out too rare and too common words.
    '''
    if rep == 'text': 
        INPUT = UNIGRAMS
    elif rep == 'user': 
        INPUT = USERS
    dfs = Counter() 
    num_sr = 0
    for sr in tqdm(os.listdir(INPUT)): 
        words = set() 
        if sr.startswith('.'): continue
        num_sr += 1
        for month in os.listdir(INPUT + sr): 
            if month.startswith('.'): continue
            with open(INPUT + sr + '/' + month, 'r') as inputfile: 
                for line in inputfile: 
                    words.add(line.split()[0])
        for w in words: 
            dfs[w] += 1
    idfs = {}
    for w in dfs: 
        if rep == 'text' and dfs[w] >= 5 and dfs[w] <= 0.95*num_sr: 
            idfs[w] = math.log10(num_sr/float(dfs[w]))
        elif rep == 'user' and dfs[w] > 1 and dfs[w] <= 0.95*num_sr: 
            # don't penalize low users but penalize bots (high)
            idfs[w] = math.log10(num_sr/float(dfs[w]))
    return idfs
    

def get_tf_idf(idfs, rep): 
    '''
    For each word in idfs, get its (1 + log tf) 
    for each document. 
    Create vectors for each document where each 
    index corresponds to a word and the value is
    (1 + log tf)xlog_10(N/df). 
    '''
    if rep == 'text': 
        INPUT = UNIGRAMS
        OUTPUT = UNI_TFIDF
        VOCAB = UNI_VOCAB
        ROWS = UNI_ROWS
    elif rep == 'user': 
        INPUT = USERS
        OUTPUT = USR_TFIDF
        VOCAB = USR_VOCAB
        ROWS = USR_ROWS
    vocab = sorted(idfs.keys())
    srs = []
    X = []
    num_sr = 0
    for sr in tqdm(os.listdir(INPUT)): 
        if sr.startswith('.'): continue
        srs.append(sr)
        tfs = Counter() 
        num_sr += 1
        for month in os.listdir(INPUT + sr): 
            if month.startswith('.'): continue
            with open(INPUT + sr + '/' + month, 'r') as inputfile: 
                for line in inputfile: 
                    contents = line.strip().split()
                    w = contents[0] 
                    if w in idfs: 
                        count = int(contents[-1])
                        tfs[w] += count
        vec = np.zeros(len(vocab))
        for i, w in enumerate(vocab): 
            if tfs[w] > 0: 
                vec[i] = (1 + math.log10(tfs[w]))*idfs[w]
        X.append(vec)
    X = np.array(X)
    print X.shape
    np.save(OUTPUT, X)
    with open(ROWS, 'w') as outputfile: 
        for sr in srs: 
            outputfile.write(sr + '\n') 
    with open(VOCAB, 'w') as outputfile: 
        for w in vocab: 
            outputfile.write(w + '\n') 
    print 1 - spatial.distance.cosine(X[srs.index('android')], X[srs.index('apple')])
    print 1 - spatial.distance.cosine(X[srs.index('london')], X[srs.index('ukpolitics')])
    print 1 - spatial.distance.cosine(X[srs.index('london')], X[srs.index('android')])
    
def svd_tf_idf(rep): 
    if rep == 'text': 
        INPUT = UNI_TFIDF
        OUTPUT = UNI_SVD_TFIDF
    elif rep == 'user': 
        INPUT = USR_TFIDF
        OUTPUT = USR_SVD_TFIDF
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    X = np.load(INPUT)
    X_new = svd.fit_transform(X)
    normalizer = Normalizer(copy=False)
    X_new_new = normalizer.fit_transform(X_new)
    np.save(OUTPUT, X_new_new)

def main(): 
    #idfs = get_idf('user') 
    #get_tf_idf(idfs, 'user')
    svd_tf_idf('user')
    svd_tf_idf('text')

if __name__ == '__main__':
    main()