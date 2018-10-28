import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.spatial.distance
from scipy.sparse import coo_matrix,save_npz,load_npz
from scipy.sparse.linalg import svds, eigs
import pyximport; pyximport.install() #I don't think this is necessary
from socialsent import seeds, util
from socialsent import lexicons
from socialsent.polarity_induction_methods import random_walk,label_propagate_probabilistic,_bootstrap_func,bootstrap
from socialsent.representations.representation_factory import create_representation
import operator
import csv
import pickle as cPickle
import glob
import sys
import os
from nltk.corpus import stopwords
from sklearn import preprocessing
from operator import itemgetter
import itertools
import pickle
import scipy


"""
Pipeline
----------
1. Download all gender-related subreddits (concatenated) - place into concat_subs 
2. Get count matrices for each subreddit in concat_subs - place into cooccur_matrices_unweighted
3. Do PPMI + SVD - place into ppmi_svd_vectors - make sure to write it as [word,v1,v2,...] line (space separated)
4. Run SocialSent on ppmi_svd_vectors - save to socialsent_lexicons_ppmi_svd 
5. Create SocialSent vector from socialsent_lexicons_ppmi_svd - save to socialsent_vectors_ppmi_svd (as npy)
"""


""" get_vocabulary
Gets word counts of subreddit. 
Assumes that all content from single subreddit is already concatenated into one .txt file, and all subreddit 
texts are in the same directory (here, referred to as concat_dir)
All text from subreddits is located within concat_dir directory with the name [subreddit].txt
Saves vocab information as txt files in vocab_dir
NOTE: vocab_count bash script is part of the GloVe package.
"""
def get_vocabulary(subreddit,concat_dir,vocab_dir):
    corpus = os.path.join(concat_dir,subreddit + '.txt')
    vocab = os.path.join(vocab_dir,subreddit + '.txt')
    os.system('./vocab_count -verbose 2 -max-vocab 100000 -min-count 5 < {} > {}'.format(corpus,vocab))



""" get_count_matrices
Gets unweighted cooccurrence matrix for each subreddit using symmetric window size of 4
Saves as .bin files to cooccur_dir 
NOTE: cooccur bash script is part of the GloVe package. 
"""
def get_count_matrices(subreddit,concat_dir,vocab_dir,cooccur_dir):
    corpus = os.path.join(concat_dir,subreddit + '.txt')
    vocab = os.path.join(vocab_dir,subreddit + '.txt')
    cooccur_matrix = os.path.join(cooccur_dir,subreddit + '.bin')
    os.system('./cooccur -verbose 2 -symmetric 1 -window-size 4 -vocab-file {} -memory 8.0 -distance-weighting 0 < {} > {}'.format(vocab,corpus,cooccur_matrix))



""" prepare_matrix_for_pmi
Called later in get_ppmi_svd()
Stores co-occurrence information into numpy matrix 
Does calculations for some parameters in PPMI calculation
    - Smoothing parameter (0 in original socialsent)
    - Context distribution scaling (when True, set to 0.75 as in original socialsent)
Returns original co-occurrence matrix (old_mat), row_probs, col_probs, and smooth parameter
NOTE: calculations involving smooth,row_probs, and col_probs are adapted from Will Hamilton's released code
"""
def prepare_matrix_for_pmi(data,num_words,smooth=0,cds=True):
    mat = np.zeros((num_words,num_words))
    for x,i in enumerate(data):
        mat[i[1]-1][i[0]-1] = i[2]
        mat[i[0]-1][i[1]-1] = i[2]
    old_mat = coo_matrix(mat).tocsr()    
    smooth = old_mat.sum() * smooth
    row_probs = old_mat.sum(1) + smooth
    col_probs = old_mat.sum(0) + smooth
    if cds:
        col_probs = np.power(col_probs, 0.75)
    row_probs = row_probs / row_probs.sum()
    col_probs = col_probs / col_probs.sum()
    print('Prepared original matrix for PPMI')
    return old_mat,row_probs,col_probs,smooth

""" make_ppmi_mat
Creates and returns PPMI matrix from original co-occurrence matrix 
Set to SocialSent default values for neg and normalize 
NOTE: PPMI calculations adapted from Will Hamilton's released SocialSent code
"""
def make_ppmi_mat(old_mat, row_probs, col_probs, smooth, neg=1, normalize=False):
    prob_norm = old_mat.sum() + (old_mat.shape[0] * old_mat.shape[1]) * smooth
    old_mat = old_mat.tocoo()
    row_d = old_mat.row
    col_d = old_mat.col
    data_d = old_mat.data
    neg = np.log(neg)
    for i in range(len(old_mat.data)):
        if data_d[i] == 0.0:
            continue
        joint_prob = (data_d[i] + smooth) / prob_norm
        denom = row_probs[row_d[i], 0] * col_probs[0, col_d[i]]
        if denom == 0.0:
            data_d[i] = 0
            continue
        data_d[i] = np.log(joint_prob /  denom)
        data_d[i] = max(data_d[i] - neg, 0)
        if normalize:
            data_d[i] /= -1*np.log(joint_prob)
    print('Made PPMI matrix')
    return coo_matrix((data_d, (row_d, col_d)))

""" get_ppmi_svd
Overarching function to run both PPMI and SVD from co-occurrence matrix 
SVD set to reduce to 100 dimensions and normalized with L2 norm (normalize set to False in header has to do with PPMI, not SVD)
Each line of output file consists of word followed by 100-dimension vector (stored as space-separated string)
Written to file named [subreddit].txt in another directory (here, called ppmi_svd_dir)
"""

def get_ppmi_svd(subreddit,vocab_dir,cooccur_dir,ppmi_svd_dir,smooth=0,cds=True,neg=1,normalize=False):
    vocab_file = os.path.join(vocab_dir,subreddit + '.txt')
    with open(vocab_file) as f:
        vocab = [v.split('\n')[0].split(' ')[0] for v in f.readlines()]

    infile = os.path.join(cooccur_dir,subreddit + '.bin')
    dt = np.dtype([('word1',np.uint32),('word2',np.uint32),('count',np.double)])
    data = np.fromfile(infile, dtype=dt) 
    old_mat,row_probs,col_probs,smooth = prepare_matrix_for_pmi(data,len(vocab),smooth,cds)
    ppmi_mat = make_ppmi_mat(old_mat, row_probs, col_probs,smooth,neg,normalize)

    u, s, vt = svds(ppmi_mat, k=100)
    l2norm = np.sqrt((u * u).sum(axis=1)).reshape(len(u),1)
    norm_vecs = u / l2norm

    outfile = os.path.join(ppmi_svd_dir,subreddit + '.txt')
    with open(outfile,'w') as f:
        for ix,w in enumerate(vocab):
            vec_list = [str(n) for n in list(norm_vecs[ix])]
            vec_string = " ".join(vec_list)
            line = w + " " + vec_string + "\n"
            f.write(line)
    print('Created PPMI-SVD vectors')


""" run_sentprop
Runs sentprop algorithm (largely adapted from released SocialSent)
Beta value of random walk set to 0.9, number of nearest neighbors set to 25 
Resulting lexicon stored in socialsent_lexicons_dir 
topn = n most frequent words for which to induce lexicons
NOTE: some code adapted from original SocialSent released code including random_walk 
"""
def run_sentprop(subreddit,ppmi_svd_dir,socialsent_lexicons_dir,vocab_dir,topn=5000,bstrp=False,nn=25,beta=0.9):
    #program = 'python make_sent_lexicons.py ' + subreddit + " " + ppmi_svd_dir + " " + socialsent_lexicons_dir + " " + vocab_dir
    #os.system(program)

    #stop_words = set(stopwords.words('english'))
    #stop_words.add('<#S#>') #dummy token 

    fname = os.path.join(vocab_dir,subreddit + '.txt')
    with open(fname,'r') as f:
        words = f.readlines()

    top_words = [w.split()[0] for w in words][:topn] 
    pos_seeds, neg_seeds = seeds.twitter_seeds() #Twitter seed words (from socialsent package)

    vector_file = os.path.join(ppmi_svd_dir,subreddit + '.txt')
    embeddings = create_representation('GIGA', vector_file,
        set(top_words).union(pos_seeds).union(neg_seeds)) # sub_vecs

    if bstrp:
        polarities = bootstrap(embeddings, pos_seeds, neg_seeds, return_all=True,
                    nn=nn, beta=beta, num_boots=50, n_procs=10) # NEW
        outfile = os.path.join(socialsent_lexicons_dir,subreddit + '.pkl') # NEW
        util.write_pickle(polarities, outfile) # NEW
    else:
        polarities = random_walk(embeddings, pos_seeds, neg_seeds, beta=beta, 
                                 nn=nn, num_boots=50, n_procs=10)
        sorted_x = sorted(polarities.items(), key=operator.itemgetter(1))
        outfile = os.path.join(socialsent_lexicons_dir,subreddit + '.txt')

        with open(outfile,'w') as f:
            tsvin = csv.writer(f,delimiter='\t')
            for word in sorted_x:
                tsvin.writerow(word)


""" make_sent_lexicon
Full pipeline to run everything for one subreddit
Specify all relevant directories 
"""
def make_sent_lexicon(subreddit,concat_dir,vocab_dir,cooccur_dir,
    ppmi_svd_dir,socialsent_lexicons_dir,socialsent_vectors_dir,bstrp):

    get_vocabulary(subreddit,concat_dir,vocab_dir)
    get_count_matrices(subreddit,concat_dir,vocab_dir,cooccur_dir)
    get_ppmi_svd(subreddit,vocab_dir,cooccur_dir,ppmi_svd_dir)
    run_sentprop(subreddit,ppmi_svd_dir,socialsent_lexicons_dir,vocab_dir,bstrp)


""" 
All functions below are for creating subreddit sentiment vectors from the lexicon
"""


"""
Normalizes sentiment lexicon to have 0 mean and unit variance (as in original paper)
Returns lexicon as a dict containing {word:normalized score} k,v pairs 
"""
def normalize_lexicon(socialsent_lexicon_dir,subreddit):
    sent_dict = {}
    scaled_sent_dict = {}
    infile_path = os.path.join(socialsent_lexicon_dir, subreddit + '.txt')
    with open(infile_path,'r') as f:
        tsvin = csv.reader(f,delimiter='\t')
        for line in tsvin:
            sent_dict[line[0]] = float(line[1])
    words_ordered = sorted(sent_dict.keys())
    sents_ordered = [sent_dict[w] for w in words_ordered]
    sents_ordered_norm = preprocessing.scale(sents_ordered)

    for i in range(len(words_ordered)):
        w = words_ordered[i]
        scaled_sent_dict[w] = float(sents_ordered_norm[i])
    return scaled_sent_dict


"""
For a specific given subreddit, returns the set of words for which sentiment was induced 
"""
def get_vocab(socialsent_lexicons_dir,subreddit):
    vocab = []
    infile_path = os.path.join(socialsent_lexicons_dir, subreddit + '.txt')
    with open(infile_path,'r') as f:
        tsvin = csv.reader(f,delimiter='\t')
        for line in tsvin:
            vocab.append(line[0])
    return set(vocab)

"""
Given the full vocabulary (set of all words that have sentiment in ANY subreddit), creates vector 
Adds 0.0 for any word that does not have a sentiment value in specific subreddit
"""
def get_vector_for_lexicon(vocab,scaled_sent_dict):
    vec_list = []
    for w in vocab:
        if w in scaled_sent_dict:
            vec_list.append(scaled_sent_dict[w])
        else:
            vec_list.append(0.0)
    return np.array(vec_list)


def read_in_bootstraped_values(socialsent_lexicons_dir,subreddit): 
    scaled_sent_dict = {}
    scaled_sent_stds = {}
    with open(os.path.join(socialsent_lexicons_dir, subreddit + '.txt'), 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            scaled_sent_dict[contents[0]] = float(contents[1])
            scaled_sent_stds[contents[0]] = float(contents[2])
    return scaled_sent_stds, scaled_sent_dict

"""
Creates sentiment vectors for all subreddits listed in subreddit_names after lexicon induction 
Saves vectors as .np files to socialsent_vectors_dir 
"""
def make_sentprop_vectors(subreddit_names,socialsent_lexicons_dir,socialsent_vectors_dir,bstrp):
    full_vocab = set()
    for subreddit in subreddit_names:
        full_vocab = full_vocab | get_vocab(socialsent_lexicons_dir,subreddit)
    vocab_ordered = sorted(full_vocab)
    with open('vocab_socialsent_ordered.pkl', "wb") as output_file:
        pickle.dump(vocab_ordered, output_file)

    for subreddit in subreddit_names:
        if bstrp: 
            scaled_sent_stds, scaled_sent_dict = read_in_bootstraped_values(socialsent_lexicons_dir,subreddit)
            # create and save std vector 
            std_vector = get_vector_for_lexicon(vocab_ordered,scaled_sent_stds)
            filename = os.path.join(socialsent_vectors_dir,subreddit + '_std.npy')
            np.save(filename,std_vector)
        else:
            scaled_sent_dict = normalize_lexicon(socialsent_lexicons_dir,subreddit)
        vector = get_vector_for_lexicon(vocab_ordered,scaled_sent_dict)
        filename = os.path.join(socialsent_vectors_dir,subreddit + '.npy')
        np.save(filename,vector)
        
def make_bootstrap_lexicon(subreddit_names, socialsent_lexicons_dir): 
    for sr in subreddit_names: 
        print sr
        with open(os.path.join(socialsent_lexicons_dir, sr + '.pkl'), 'r') as f:
            item = cPickle.load(f)
        kys = sorted(item[0].keys())
        polarities = {} # word : mean
        stds = {} # word : std
        # normalize each dictionary to have zero mean and unit variance
        normal_item = [] # list of lists of subreddit-scaled values
        for i in range(len(item)):
            one_iter = []
            for k in kys: 
                one_iter.append(item[i][k])
            normal_lexicon = preprocessing.scale(one_iter)
            normal_item.append(normal_lexicon)
        # then get mean and std 
        for k in kys: 
            vals = []
            for i in range(len(item)):
                vals.append(normal_item[i][kys.index(k)])
            polarities[k] = np.mean(vals)
            stds[k] = np.std(vals)
        sorted_pols = sorted(polarities.items(), key=operator.itemgetter(1))
                  
        with open(os.path.join(socialsent_lexicons_dir, sr + '.txt'), 'w') as outfile:
            for p in sorted_pols: 
                outfile.write(p[0] + '\t' + str(p[1]) + '\t' + str(stds[p[0]]) + '\n')
                
def modify_parameters(): 
    subreddit_list  = ['femalefashionadvice','malefashionadvice','mensrights',
                       'trollxchromosomes','actuallesbians','askmen','askwomen','askgaybros','xxfitness']
    ppmi_svd_dir = '../logs/ppmi_svd_vectors'
    sent_lexicons_dir_main = '../logs/socialsent_lexicons_ppmi_svd_top5000'
    vocab_dir = '../logs/vocab_counts'
    bstrp = True
    #for params in [(0.9, 20), (0.9, 30), (0.5, 25), (0.7, 25)]: 
    for params in [(0.9, 15), (0.9, 35)]:
        beta, nn = params
        sent_lexicons_dir = sent_lexicons_dir_main + '_' + str(nn) + '_' + str(beta)
        print sent_lexicons_dir
        if not os.path.exists(sent_lexicons_dir):
            os.makedirs(sent_lexicons_dir)
        for subreddit in subreddit_list:
            run_sentprop(subreddit, ppmi_svd_dir, sent_lexicons_dir, vocab_dir, topn=5000, bstrp=bstrp, nn=nn,beta=beta)
        if bstrp: 
            make_bootstrap_lexicon(subreddit_list, sent_lexicons_dir)
        
def run_main_pipeline(): 
    bstrp = True
    vocab_dir = '../logs/vocab_counts'
    concat_dir = '../logs/concat_subs'
    cooccur_dir = '../logs/cooccur_matrices_unweighted'
    ppmi_svd_dir = '../logs/ppmi_svd_vectors'
    sent_lexicons_dir = '../logs/socialsent_lexicons_ppmi_svd_top5000'
    sent_vectors_dir = '../logs/socialsent_vectors_ppmi_svd_top5000'

    #all gender subreddits ('london' no longer included)
    subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights',
    'trollxchromosomes','actuallesbians','askmen','askwomen','askgaybros','xxfitness']

    #makes sentiment lexicons for each subreddit
    for subreddit in subreddit_list:
        print(subreddit)
        make_sent_lexicon(subreddit,concat_dir,vocab_dir,cooccur_dir,ppmi_svd_dir,sent_lexicons_dir,sent_vectors_dir, bstrp)
        
    if bstrp: 
        make_bootstrap_lexicon(subreddit_list, sent_lexicons_dir)

    #makes vectors for each subreddit after lexicon induction
    make_sentprop_vectors(subreddit_list,sent_lexicons_dir,sent_vectors_dir,bstrp)

def main():
    modify_parameters()

if __name__ == "__main__":
	main()
