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
SR_LIST = '/dfs/scratch2/lucy3/reddit-sent/data/listofsubreddits.txt'

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
import codecs

def tsne_viz(df, vocab, colors=None, output_filename=None):
    """  
    Modified from Chris Potts's implementation at 
    https://github.com/cgpotts/cs224u/blob/master/vsm.py
    """
    # Recommended reduction via PCA or similar:
    n_components = 50 if df.shape[1] >= 50 else df.shape[1]
    dimreduce = PCA(n_components=n_components)
    X = dimreduce.fit_transform(df)
    # t-SNE:
    tsne = TSNE(n_components=2, random_state=0)
    tsnemat = tsne.fit_transform(X)
    # Plot values:
    xvals = tsnemat[: , 0]
    yvals = tsnemat[: , 1]
    # Plotting:
    plt.scatter(xvals, yvals, c=colors)
    plt.title('2d plot of subreddits using t-SNE, n = ' + str(len(set(colors))))
    # Output:
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    
def cluster_plot(rep, write=False, plot=False, k=20): 
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
    kmeans = AgglomerativeClustering(n_clusters=k).fit(X)
    labels = defaultdict(set)
    for i in range(len(kmeans.labels_)): 
        label = kmeans.labels_[i]
        sr = srs[i] 
        labels[label].add(sr)
    if plot:
        tsne_viz(X, srs, \
                 colors=kmeans.labels_, \
                 output_filename="/dfs/scratch2/lucy3/reddit-sent/logs/" \
                 + 'tsne_' + rep + '.png')
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

def topic(): 
    srs = []
    with open(UNI_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            srs.append(line.strip())
    reddits = defaultdict(set)
    curr = 0
    with open(SR_LIST, 'r') as inputfile: 
        for line in inputfile: 
            if line.startswith('/r/'): 
                sr = line.strip()[3:].lower()
                if sr in srs: 
                    reddits[curr].add(sr)
            elif line.strip() == 'General Content' or \
                line.strip() == 'Discussion' or \
                line.strip() == 'Educational' or \
                line.strip() == 'Entertainment' or \
                line.strip() == 'Hobbies/Occupations' or \
                line.strip() == 'Lifestyle' or \
                line.strip() == 'Technology' or \
                line.strip() == 'Humor' or \
                line.strip() == 'Other':
                    curr += 1
    return reddits

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
    '''
    Perhaps I should use mutual information
    rather than purity. 
    '''
    topic_labels = topic()
    text_labels = cluster_plot('text', plot=True, write=True) 
    user_labels = cluster_plot('user', plot=True, write=True) 
    #baseline_labels = baseline()
    #print purity(text_labels, baseline_labels)
    #print purity(text_labels, user_labels)
    '''
    text_labels = cluster_plot('text', k=9) 
    user_labels = cluster_plot('user', k=9) 
    print purity(text_labels, topic_labels)
    print purity(topic_labels, text_labels)
    print purity(user_labels, topic_labels)
    print purity(topic_labels, user_labels)
    '''

if __name__ == '__main__':
    main()