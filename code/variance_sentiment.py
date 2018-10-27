import pickle 
import numpy as np
import pandas as pd 
import os
import csv
from nltk.corpus import wordnet as wn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from collections import defaultdict
    
def create_dataframe(vector_dir,subreddit_list):
    vecs = {}
    for sub in subreddit_list:
        vec_file = os.path.join(vector_dir,sub + '.npy')
        vec = np.load(vec_file)
        vecs[sub] = vec
    df = pd.DataFrame(vecs)
    return df

def get_variances(df):
    variances = df.var(axis=1)
    idx = list(np.argsort(variances, axis=1))
    return idx,variances

def calculate_variances(): 
    vector_dir = '../logs/socialsent_vectors_ppmi_svd_top5000'
    subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes','actuallesbians','askmen','askwomen','askgaybros','xxfitness']
    vocab_file = 'vocab_socialsent_ordered.pkl'
    variance_file = 'variance_socialsent.tsv'
    df = create_dataframe(vector_dir,subreddit_list)
    idx,variances = get_variances(df)


    with open(vocab_file,'r') as f:
        vocab_list = pickle.load(f)

    for num,i in enumerate(idx[-30:][::-1]):
        print num,vocab_list[i]
        print '--'
        print df.loc[i]
        print '-----'

    with open('../logs/' + variance_file,'w') as tsvin:
        tsvin = csv.writer(tsvin,delimiter='\t')
        for i in idx:
            tsvin.writerow((vocab_list[i],variances[i]))

def variance_vs_neighbors(): 
    # hypothesis: high variance due to having lots of diff meanings
    # diff meanings -> lots of diff neighbors across subreddits
    vocab_dir = '../logs/vocab_counts'
    subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes','actuallesbians','askmen','askwomen','askgaybros','xxfitness']
    variance_file = 'variance_socialsent.tsv'
    variance_dict = {}
    with open('../logs/' + variance_file, 'r') as tsvin: 
        for line in tsvin: 
            contents = line.strip().split('\t') 
            variance_dict[contents[0]] = float(contents[1])
    word_vectors = '../logs/ppmi_svd_vectors/' 
    neighbors = defaultdict(set) # for each word, union of all neighbors across subreddits
    for subreddit in subreddit_list: 
        X = []
        words = []
        vocab = {}
        print subreddit
        with open(word_vectors + subreddit + '.txt') as infile: 
            for line in infile: 
                contents = line.strip().split()
                if contents[0] not in variance_dict: 
                    # these words are just seed words 
                    pass
                words.append(contents[0])
                vocab[contents[0]] = np.array(map(float, contents[1:]), dtype='float32')
                if len(words) > 5000: 
                    break
        X = np.vstack(vocab[word] for word in words)
        X_sim = cosine_similarity(X)
        X_rank = np.argsort(X_sim, axis=1)
        for i in range(X.shape[0]): 
            for j in range(2, 22): # skip 1 because it's itself
                neighbors[words[i]].add(words[X_rank[i][-j]])
                if words[i] == 'sick':
                    print words[X_rank[i][-j]]
    print neighbors['jealous']
    print neighbors['sick']

def main():
    #calculate_variances()
    variance_vs_neighbors()



if __name__ == "__main__":
	main()
