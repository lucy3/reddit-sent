"""
kmeans clustering to confirm that
the subreddit vectors we have are
reasonable. 
"""
UNI_INPUT = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_unigrams.npy'
UNI_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/unigram_rows'
UNI_CLUST = '/dfs/scratch2/lucy3/reddit-sent/logs/unigram_clusters'
USR_INPUT = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_users.npy'
USR_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/users_rows'
USR_CLUST = '/dfs/scratch2/lucy3/reddit-sent/logs/users_clusters'

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from scipy.stats import mode
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import random
    
def cluster_plot(rep, write=False, plot=False): 
    """
    Write out text-based and user-based clusters. 
    """
    if rep == 'text': 
        ROWS = UNI_ROWS
        INPUT = UNI_INPUT
        CLUST = UNI_CLUST
    elif rep == 'user': 
        ROWS = USR_ROWS
        INPUT = USR_INPUT
        CLUST = USR_CLUST
    srs = []
    with open(ROWS, 'r') as inputfile: 
        for line in inputfile: 
            srs.append(line.strip())
    X = np.load(INPUT)
    kmeans = AgglomerativeClustering(n_clusters=20).fit(X)
    labels = defaultdict(set)
    for i in range(len(kmeans.labels_)): 
        label = kmeans.labels_[i]
        sr = srs[i] 
        labels[label].add(sr)
    if plot:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        pca1 = pca_result[:,0]
        pca2 = pca_result[:,1] 
        fig, ax = plt.subplots()
        for i in range(pca1.shape[0]): 
            ax.scatter(pca1[i], pca2[i], label=str(kmeans.labels_[i]), s=4)
            ax.annotate(srs[i], xy=(pca1[i], pca2[i]), size=5)
        plt.savefig("/dfs/scratch2/lucy3/reddit-sent/logs/" + 'pca_' + rep + '.png')
    if write: 
        with open(CLUST, 'w') as outputfile: 
            for label in labels: 
                outputfile.write(str(label) + '\t')
                for sr in labels[label]: 
                    outputfile.write(sr + ' ')
                outputfile.write('\n')
    return labels
    
def purity(text_labels, user_labels): 
    '''
    RUN MULTIPLE CLUSTERING AND GET AVERAGE
    '''
    rev_text_labels = {}
    for cluster in text_labels: 
        for sr in text_labels[cluster]:
            rev_text_labels[sr] = cluster
    s = 0
    for cluster in user_labels: 
        lbls = []
        for sr in user_labels[cluster]: 
            lbls.append(rev_text_labels[sr])
        values, counts = np.unique(lbls, return_counts=True)
        m = counts.argmax()
        s += counts[m]
    return s/float(len(rev_text_labels))

def baseline(): 
    labels = defaultdict(set)
    srs = []
    with open(UNI_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            srs.append(line.strip())
    for sr in srs: 
        labels[random.randint(0, 19)].add(sr)
    return labels

def main(): 
    text_labels = cluster_plot('text', plot=True, write=True) 
    user_labels = cluster_plot('user', plot=True, write=True) 
    baseline_labels = baseline()
    print purity(text_labels, baseline_labels)

if __name__ == '__main__':
    main()