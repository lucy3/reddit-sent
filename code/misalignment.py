'''
Misalignment evaluations
'''
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from scipy.stats import mode, spearmanr
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

UNI_INPUT = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_unigrams.npy'
UNI_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/unigram_rows'
USR_INPUT = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_users.npy'
USR_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/users_rows'
OUTPUT = '/dfs/scratch2/lucy3/reddit-sent/logs/z_score_unigrams_users.npy'
COM_COUNTS = '/dfs/scratch2/lucy3/reddit-sent/logs/comment_counts.json'

def sim_corr(): 
    uni_srs = []
    with open(UNI_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            uni_srs.append(line.strip())
    usr_srs = []
    with open(USR_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            usr_srs.append(line.strip())
    uni_srs = np.array(uni_srs, dtype=object)
    usr_srs = np.array(usr_srs, dtype=object)
    uni_sort = np.argsort(uni_srs) 
    usr_sort = np.argsort(usr_srs)
    uni_srs = uni_srs[uni_sort]
    usr_srs = usr_srs[usr_sort]
    X = np.load(UNI_INPUT)
    Y = np.load(USR_INPUT)
    X = X[uni_sort]
    Y = Y[usr_sort]
    X_vals = []
    Y_vals = []
    for i in range(X.shape[0]): 
        for j in range(i + 1, X.shape[0]): 
            text_sim = 1-cosine(X[i], X[j])
            user_sim = 1-cosine(Y[i], Y[j])
            if text_sim < 0.2 and user_sim > 0.8: 
                print "LOW TEXT, HIGH USER", uni_srs[i], uni_srs[j], usr_srs[i], usr_srs[j]
            if text_sim > 0.8 and user_sim < 0.2: 
                print "HIGH TEXT, LOW USER", uni_srs[i], uni_srs[j]
            X_vals.append(text_sim)
            Y_vals.append(user_sim)
    print "Spearman:", spearmanr(X_vals, Y_vals)
    plt.scatter(X_vals, Y_vals, alpha=0.2, s=4)
    #plt.scatter(X_vals_gen, Y_vals_gen, s=5)
    plt.xlabel('text similarity')
    plt.ylabel('user similarity')
    plt.savefig("/dfs/scratch2/lucy3/reddit-sent/logs/" + 'unigram_user_corr.png')
    
def z_score(): 
    uni_srs = []
    with open(UNI_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            uni_srs.append(line.strip())
    usr_srs = []
    with open(USR_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            usr_srs.append(line.strip())
    uni_srs = np.array(uni_srs, dtype=object)
    usr_srs = np.array(usr_srs, dtype=object)
    uni_sort = np.argsort(uni_srs) 
    usr_sort = np.argsort(usr_srs)
    uni_srs = uni_srs[uni_sort]
    usr_srs = usr_srs[usr_sort]
    X = np.load(UNI_INPUT)
    Y = np.load(USR_INPUT)
    X = X[uni_sort]
    Y = Y[usr_sort]
    X_sim = cosine_similarity(X)
    Y_sim = cosine_similarity(Y)
    # Create ranked similarity matrices
    X_rank = np.argsort(X_sim, axis=1)
    Y_rank = np.argsort(Y_sim, axis=1)
    # Create D = A - T. 
    D = X_rank - Y_rank # unigram - user
    # Standardize scores in each row by subtracting mean and dividing by standard deviation
    D = (D - np.mean(D, axis=1))/np.std(D, axis=1)
    # Standardize scores in each column
    D = (D - np.mean(D, axis=0))/np.std(D, axis=0)
    np.save(OUTPUT, D)
    
def size_sim(rep): 
    """
    Correlation between similarity and subreddit size. 
    """
    if rep == 'text': 
        ROWS = UNI_ROWS
        INPUT = UNI_INPUT
    elif rep == 'user': 
        ROWS = USR_ROWS
        INPUT = USR_INPUT
    X = np.load(INPUT)
    X_sim = cosine_similarity(X)
    X_avg_sim = np.mean(X_sim, axis=1)
    with open(COM_COUNTS, 'r') as inputfile: 
        counts = json.load(inputfile)
    X_counts = []
    with open(ROWS, 'r') as inputfile: 
        for line in inputfile: 
            X_counts.append(counts[line.strip()])
    X_counts = np.array(X_counts)
    plt.scatter(X_avg_sim, X_counts, alpha=0.2, s=4)
    plt.yscale('log')
    plt.xlabel('average similarity')
    plt.ylabel('counts')
    plt.savefig("/dfs/scratch2/lucy3/reddit-sent/logs/" + rep + '_size_sim.png')

def main():
    sim_corr()
    #z_score()

if __name__ == '__main__':
    main()