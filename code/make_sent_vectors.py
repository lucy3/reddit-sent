import csv
from sklearn import preprocessing
from operator import itemgetter
import itertools
import numpy as np
import glob
import pickle
import os
import scipy
from sklearn.decomposition import TruncatedSVD
import csv

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

def get_vocab(socialsent_lexicons_dir,subreddit):
    vocab = []
    infile_path = os.path.join(socialsent_lexicons_dir, subreddit + '.txt')
    with open(infile_path,'r') as f:
        tsvin = csv.reader(f,delimiter='\t')
        for line in tsvin:
            vocab.append(line[0])
    return set(vocab)

def get_vector_for_lexicon(vocab,scaled_sent_dict):
    vec_list = []
    for w in vocab:
        if w in scaled_sent_dict:
            vec_list.append(scaled_sent_dict[w])
        else:
            vec_list.append(0.0)
    return np.array(vec_list)


def create_raw_vectors(subreddit_names,socialsent_lexicons_dir,socialsent_vectors_dir):
    full_vocab = set()
    for subreddit in subreddit_names:
        full_vocab = full_vocab | get_vocab(socialsent_lexicons_dir,subreddit)
    vocab_ordered = sorted(full_vocab)
    with open('vocab_socialsent_ordered.pkl', "wb") as output_file:
        pickle.dump(vocab_ordered, output_file)

    for subreddit in subreddit_names:
        scaled_sent_dict = normalize_lexicon(socialsent_lexicons_dir,subreddit)
        vector = get_vector_for_lexicon(vocab_ordered,scaled_sent_dict)
        filename = os.path.join(socialsent_vectors_dir,subreddit + '.npy')
        np.save(filename,vector)


def get_vector(socialsent_vectors_dir,subreddit,rep_type):
    if rep_type == 'sent':
        sub_file = os.path.join(socialsent_vectors_dir,subreddit+ '.npy')
        print sub_file
        return np.load(sub_file)

    elif rep_type == 'text':
        UNIGRAMS_ROWS = 'unigram_rows'
        UNIGRAMS_VECS = 'svd-tf-idf_unigrams.npy'
        unigrams = []
        with open(UNIGRAMS_ROWS,'r') as inputfile:
            for line in inputfile:
                unigrams.append(line.strip())
        X_unigrams = np.load(UNIGRAMS_VECS)
        vec = X_unigrams[unigrams.index(subreddit)]
        return vec

    elif rep_type == 'user': 
        USERS_ROWS = 'users_rows'
        USERS_VECS = 'svd-tf-idf_users.npy'
        users = []
        with open(USERS_ROWS,'r') as inputfile:
            for line in inputfile:
                users.append(line.strip())
        X_users = np.load(USERS_VECS)
        vec = X_users[users.index(subreddit)]
        return vec
    else:
        return 0

def calculate_similarity(socialsent_vectors_dir,subreddit1,subreddit2,rep_type):
    sub1_vec = get_vector(socialsent_vectors_dir,subreddit1,rep_type)
    sub2_vec = get_vector(socialsent_vectors_dir,subreddit2,rep_type)

    cossim = 1 - scipy.spatial.distance.cosine(sub1_vec,sub2_vec)
    return subreddit1,subreddit2,cossim


def get_all_sims(socialsent_vectors_dir,subreddit_names,sim_dir,rep_type):
    results = []
    combinations = list(itertools.combinations(subreddit_names, 2))
    for (s1,s2) in combinations:
        elem = calculate_similarity(socialsent_vectors_dir,s1,s2,rep_type)
        results.append(list(elem))
    sorted_results = sorted(results)

    outfile = os.path.join(sim_dir,rep_type + '.csv')
    with open(outfile,'w') as f:
        csvin = csv.writer(f)
        for elem in sorted_results:
            csvin.writerow(elem)




def main():
    #concat_docs = glob.glob('concat_subs/*.txt')
    #subreddit_names = [os.path.basename(d)[:-4] for d in concat_docs
    subreddit_names = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes','actuallesbians','askmen','askwomen','askgaybros','xxfitness']
    socialsent_lexicons_dir = '../logs/socialsent_lexicons_ppmi_svd_top5000'
    socialsent_vectors_dir = '../logs/socialsent_vectors_ppmi_svd_top5000'
    sim_dir = '../logs/similarities'
    #create_raw_vectors(subreddit_names,socialsent_lexicons_dir,socialsent_vectors_dir)
    get_all_sims(socialsent_vectors_dir,subreddit_names,sim_dir,'sent')




if __name__ == "__main__":
	main()
