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
from scipy.stats import pearsonr
    
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
    '''
    Finding the words with highest variance. 
    '''
    vector_dir = '../logs/socialsent_vectors_ppmi_svd_top5000'
    subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes',
                      'actuallesbians','askmen','askwomen','askgaybros','xxfitness']
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
    '''
    Why do certain words have very high variance? 
    -> the neighbors SentProp encounters have lots of different sentiment.
    You have to be polysemous and your senses need to be very positive or negative.
    '''
    vocab_dir = '../logs/vocab_counts'
    subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes',
                      'actuallesbians','askmen','askwomen','askgaybros','xxfitness']
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

def variance_across_params(): 
    '''
    If we adjust K (number of neighbors) and beta, 
    what happens? 
    '''
    sent_lexicons_dir_main = '../logs/socialsent_lexicons_ppmi_svd_top5000'
    subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes',
                      'actuallesbians','askmen','askwomen','askgaybros','xxfitness']
    for subreddit in subreddit_list: 
        vocab = []
        default_scores = []
        with open(sent_lexicons_dir_main + '/' + subreddit + '.txt', 'r') as infile: 
            for line in infile: 
                contents = line.strip().split()
                vocab.append(contents[0])
                default_scores.append(float(contents[1]))
        #for params in [(0.5, 25), (0.7, 25), (0.9, 20), (0.9, 30)]:
        for params in [(0.9, 15), (0.9, 35)]:
            param_scores = np.zeros(len(default_scores))
            beta, nn = params
            file = sent_lexicons_dir_main + '_' + str(nn) + '_' + str(beta) + '/' + subreddit + '.txt'
            with open(file, 'r') as infile: 
                for line in infile: 
                    contents = line.strip().split()
                    idx = vocab.index(contents[0])
                    param_scores[idx] = float(contents[1])
            print subreddit, params, pearsonr(default_scores, param_scores)
            plt.scatter(default_scores, param_scores)
            plt.xlabel('default params K=25, beta=0.9')
            plt.ylabel('new params K='+str(nn)+', beta='+str(beta))
            plt.xlim(min(min(default_scores), min(param_scores)), max(max(default_scores), max(param_scores)))
            plt.ylim(min(min(default_scores), min(param_scores)), max(max(default_scores), max(param_scores))) 
            plt.title(subreddit)
            plt.savefig('../logs/parameter_comparison/' + str(nn) + '_' + str(beta) + '_' + subreddit + '.png')
            plt.close()

def main():
    #calculate_variances()
    variance_vs_neighbors()
    #variance_across_params()



if __name__ == "__main__":
	main()
