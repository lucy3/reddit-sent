"""
kmeans clustering to confirm that
the subreddit vectors we have are
reasonable. 
"""
INPUT = '../logs/tf-idf_unigrams.npy'
SUBREDS = '../logs/tf-idf_rows'

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

def nearest_neighbors(): 
    """
    NOT FINISHED YET
    
    TODO: SVD on TF-IDF
    """
    return 
    X = np.load(INPUT)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(X)
    distances, indices = knn_model.kneighbors(query_tf_idf, n_neighbors = k+1)
    
def purity_bcubed(): 
    """
    See how well the clusters match up. 
    
    NOT FINISHED YET
    
    TODO: SVD on TF-IDF
    """
    srs = []
    with open(SUBREDS, 'r') as inputfile: 
        for line in inputfile: 
            srs.append(line.strip())
    X = np.load(INPUT)
    kmeans = KMeans(n_clusters=40, random_state=0).fit(X)
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

def main(): 
    #nearest_neighbors() 
    purity_bcubed() 

if __name__ == '__main__':
    main()