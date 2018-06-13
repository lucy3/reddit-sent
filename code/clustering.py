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
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    silhouette_score, silhouette_samples
import matplotlib.cm as cm
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

def tsne_viz(df, vocab, rep, colors=None, output_filename=None):
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
    colormap = plt.cm.nipy_spectral
    plt.rcParams['axes.color_cycle'] = [colormap(i) for i in np.linspace(0, 1,len(set(colors)))]
    # Plotting:
    for i in set(colors): 
        x = [p for (j,p) in enumerate(xvals) if colors[j]==i]
        y = [p for (j,p) in enumerate(yvals) if colors[j]==i]
        plt.scatter(x, y, label=str(i), alpha=0.8, s=6)
    plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5), ncol=2)
    plt.title(rep.title() + '-based Clusters, n = ' + str(len(set(colors))))
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
    # sort by subreddit 
    srs = np.array(srs, dtype=object)
    srs_sort = np.argsort(srs) 
    srs = srs[srs_sort]
    X= X[srs_sort]
    # cluster 
    kmeans = AgglomerativeClustering(n_clusters=k).fit(X)
    labels_dict = defaultdict(set)
    for i in range(len(kmeans.labels_)): 
        label = kmeans.labels_[i]
        sr = srs[i] 
        labels_dict[label].add(sr)
    if plot:
        tsne_viz(X, srs, rep, \
                 colors=kmeans.labels_, \
                 output_filename="/dfs/scratch2/lucy3/reddit-sent/logs/" \
                 + 'tsne_' + rep + '.png')
    if write: 
        with open(CLUST, 'w') as outputfile: 
            for label in labels_dict: 
                outputfile.write(str(label) + '\t')
                for sr in labels_dict[label]: 
                    outputfile.write(sr + ' ')
                outputfile.write('\n')
    return labels_dict, kmeans.labels_, X
    
def purity(text_labels, user_labels): 
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

def topic(many=False): 
    srs = []
    with open(UNI_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            srs.append(line.strip())
    reddits = defaultdict(set)
    srs = sorted(srs)
    labels = np.zeros(len(srs))
    curr = 0
    with open(SR_LIST, 'r') as inputfile: 
        for line in inputfile: 
            if line.startswith('/r/'): 
                sr = line.strip()[3:].lower()
                if sr in srs: 
                    labels[srs.index(sr)] = curr
                    reddits[curr].add(sr)
            elif many and not line.startswith('$') and line.strip() != '': 
                curr += 1
            elif not many and line.startswith('$'): 
                curr += 1
    return reddits, labels

def baseline(n=20): 
    labels_dict = defaultdict(set)
    srs = []
    with open(UNI_ROWS, 'r') as inputfile: 
        for line in inputfile: 
            srs.append(line.strip())
    srs = sorted(srs)
    labels = []
    for sr in srs: 
        cluster = random.randint(0, n-1)
        labels_dict[cluster].add(sr)
        labels.append(cluster)
    return labels_dict, labels

def evaluation(): 
    rand_text_topic = []
    rand_text_user = []
    rand_text_random = []
    rand_user_topic = []
    nmi_text_user = []
    nmi_text_topic = []
    nmi_text_random = []
    nmi_user_topic = []
    ks = [5, 9, 13, 17, 20, 21, 25, 28, 30, 32, 35]
    for k in ks: 
        print "K=", k
        text_labels_dict, text_labels, text_X = cluster_plot('text', k=k) 
        user_labels_dict, user_labels, user_X = cluster_plot('user', k=k) 
        baseline_labels_dict, baseline_labels = baseline(n=k)
        topic_labels_dict, topic_labels = topic()
        rand_text_topic.append(adjusted_rand_score(text_labels, topic_labels))
        rand_text_user.append(adjusted_rand_score(text_labels, user_labels))
        rand_text_random.append(adjusted_rand_score(text_labels, baseline_labels))
        rand_user_topic.append(adjusted_rand_score(user_labels, topic_labels))
        nmi_text_user.append(adjusted_mutual_info_score(text_labels, user_labels))
        nmi_text_topic.append(adjusted_mutual_info_score(text_labels, topic_labels))
        nmi_text_random.append(adjusted_mutual_info_score(text_labels, baseline_labels))
        nmi_user_topic.append(adjusted_mutual_info_score(user_labels, topic_labels))
    plt.plot(ks, rand_text_topic, label="rand_text_topic")
    plt.plot(ks, rand_text_user, label="rand_text_user")
    plt.plot(ks, rand_text_random, label="rand_text_random")
    plt.plot(ks, rand_user_topic, label="rand_user_topic")
    plt.plot(ks, nmi_text_user, label="nmi_text_user")
    plt.plot(ks, nmi_text_topic, label="nmi_text_topic")
    plt.plot(ks, nmi_text_random, label="nmi_text_random")
    plt.plot(ks, nmi_user_topic, label="nmi_user_topic")
    plt.legend()
    plt.savefig("../logs/evaluation.png")

def choosing_k(rep): 
    '''
    https://stackoverflow.com/questions/37767298/another-function-useful-than-elbow-in-finding-k-clusters
    '''
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
    # sort by subreddit 
    srs = np.array(srs, dtype=object)
    srs_sort = np.argsort(srs) 
    srs = srs[srs_sort]
    X= X[srs_sort]
    # cluster 
    Ks = range(2, 100)
    km = [AgglomerativeClustering(n_clusters=i) for i in Ks]
    scores = [silhouette_score(X, km[i].fit(X).labels_) for i in range(len(km))]
    plt.plot(Ks, scores)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Choosing optimal k')
    plt.savefig('../logs/choosing_k'+rep+'.png')
    plt.close()
    
def choosing_k_plots(rep):
    """
    Pulled from 
    http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
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
    # sort by subreddit 
    srs = np.array(srs, dtype=object)
    srs_sort = np.argsort(srs) 
    srs = srs[srs_sort]
    X= X[srs_sort]
    # cluster 
    Ks = range(2, 100)
    for n_clusters in Ks: 
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        ax = plt.gca()
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.savefig('../logs/ChoosingK/silhouette_'+str(n_clusters)+rep+'.png')
        plt.close()

def main():
    evaluation() 
    #choosing_k('text')
    #choosing_k('user')

if __name__ == '__main__':
    main()
