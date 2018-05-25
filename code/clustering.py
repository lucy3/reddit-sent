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
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
    
def cluster(rep): 
    """
    See how well the clusters match up. 
    
    NOT FINISHED YET
    
    TODO: SVD on TF-IDF
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
    with open(CLUST, 'w') as outputfile: 
        for label in labels: 
            outputfile.write(str(label) + '\t')
            for sr in labels[label]: 
                outputfile.write(sr + ' ')
            outputfile.write('\n')
        
def pca(rep): 
    if rep == 'text': 
        ROWS = UNI_ROWS
        INPUT = UNI_INPUT
    elif rep == 'user': 
        ROWS = USR_ROWS
        INPUT = USR_INPUT
    srs = []
    with open(ROWS, 'r') as inputfile: 
        for line in inputfile: 
            srs.append(line.strip())
    X = np.load(INPUT)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    pca1 = pca_result[:,0]
    pca2 = pca_result[:,1] 
    fig, ax = plt.subplots()
    ax.scatter(pca1, pca2)
    for i, txt in enumerate(srs):
        ax.annotate(txt, (pca1[i],pca2[i]))
    plt.savefig("/dfs/scratch2/lucy3/reddit-sent/logs/" + 'pca_' + rep + '.png')

def main(): 
    cluster('text') 
    cluster('user') 
    #pca('text')
    #pca('user')

if __name__ == '__main__':
    main()