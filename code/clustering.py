"""
kmeans clustering to confirm that
the subreddit vectors we have are
reasonable. 
"""
UNI_INPUT = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_unigrams.npy'
UNI_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/unigram_rows'
USR_INPUT = '/dfs/scratch2/lucy3/reddit-sent/logs/svd-tf-idf_users.npy'
USR_ROWS = '/dfs/scratch2/lucy3/reddit-sent/logs/users_rows'

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict
from ggplot import *
    
def cluster(rep): 
    """
    See how well the clusters match up. 
    
    NOT FINISHED YET
    
    TODO: SVD on TF-IDF
    """
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
    kmeans = AgglomerativeClustering(n_clusters=20).fit(X)
    labels = defaultdict(set)
    for i in range(len(kmeans.labels_)): 
        label = kmeans.labels_[i]
        sr = srs[i] 
        labels[label].add(sr)
    for label in labels: 
        print label, '\t', 
        for sr in labels[label]: 
            print sr,
        print
        
def pca(rep): 
    if rep == 'text': 
        ROWS = UNI_ROWS
        INPUT = UNI_INPUT
    elif rep == 'user': 
        ROWS = USR_ROWS
        INPUT = USR_INPUT
    X = np.load(INPUT)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
    ggsave(plot = chart, filename = 'pca_' + rep + '.png', path = "/dfs/scratch2/lucy3/reddit-sent/logs/")

def main(): 
    #cluster('text') 
    pca('text')

if __name__ == '__main__':
    main()