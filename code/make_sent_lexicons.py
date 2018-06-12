from socialsent import seeds
from socialsent import lexicons
from socialsent.polarity_induction_methods import random_walk,label_propagate_probabilistic,_bootstrap_func,bootstrap
#from socialsent.evaluate_methods import binary_metrics
from socialsent.representations.representation_factory import create_representation
import operator
import csv
import glob
import sys
import os
from nltk.corpus import stopwords


# def _bootstrap_func(embeddings, positive_seeds, negative_seeds, boot_size, score_method, seed, **kwargs):

# def bootstrap(embeddings, positive_seeds, negative_seeds, num_boots=10, score_method=random_walk,
#         boot_size=7, return_all=False, n_procs=15, **kwargs):



if __name__ == "__main__":
    subreddit = sys.argv[1]
    vector_dir = sys.argv[2]
    sent_lexicon_dir = sys.argv[3]
    vocab_dir = sys.argv[4]
    stop_words = set(stopwords.words('english'))
    stop_words.add('<#S#>')

    fname = os.path.join(vocab_dir,subreddit + '.txt')
    with open(fname,'r') as f:
        words = f.readlines()

    top_5000 = [w.split()[0] for w in words if w not in stop_words][:5000]

    pos_seeds, neg_seeds = seeds.twitter_seeds() #Twitter seed words
    vector_file = os.path.join(vector_dir,subreddit + '.txt')
    embeddings = create_representation('GIGA', vector_file,
        set(top_5000).union(pos_seeds).union(neg_seeds))

    polarities = bootstrap(embeddings, pos_seeds, neg_seeds, return_all=True,
                nn=25, beta=0.9, num_boots=2, n_procs=10)
    print polarities[0]
   
    # polarities = random_walk(embeddings, pos_seeds, neg_seeds, beta=0.9, nn=25,
    #         num_boots=50,n_procs=10)
    sorted_x = sorted(polarities.items(), key=operator.itemgetter(1))
    outfile = os.path.join(sent_lexicon_dir,subreddit + '.txt')

    with open(outfile,'w') as f:
        tsvin = csv.writer(f,delimiter='\t')
        for word in sorted_x:
            tsvin.writerow(word)

   

