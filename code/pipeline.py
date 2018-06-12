import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.spatial.distance
from scipy.sparse import coo_matrix,save_npz,load_npz
from scipy.sparse.linalg import svds, eigs
import pyximport; pyximport.install()
import os


"""
Pipeline
----------
1. Download all gender-related subreddits (concatenated) - place into concat_subs 
2. Get count matrices for each subreddit in concat_subs - place into cooccur_matrices_unweighted
3. Do PPMI + SVD - place into ppmi_svd_vectors - make sure to write it as [word,v1,v2,...] line (space separated)
4. Run SocialSent on ppmi_svd_vectors - save to socialsent_lexicons_ppmi_svd 
5. Create SocialSent vector from socialsent_lexicons_ppmi_svd - save to socialsent_vectors_ppmi_svd (as npy)


separately download everything
make expicit list of subreddit names
run loop on each subreddit name 
"""

def get_vocabulary(subreddit,concat_dir,vocab_dir):
	corpus = os.path.join(concat_dir,subreddit + '.txt')
	vocab = os.path.join(vocab_dir,subreddit + '.txt')
	os.system('./vocab_count -verbose 2 -max-vocab 100000 -min-count 5 < {} > {}'.format(corpus,vocab))

def get_count_matrices(subreddit,concat_dir,vocab_dir,cooccur_dir):
	corpus = os.path.join(concat_dir,subreddit + '.txt')
	vocab = os.path.join(vocab_dir,subreddit + '.txt')
	cooccur_matrix = os.path.join(cooccur_dir,subreddit + '.bin')
	os.system('./cooccur -verbose 2 -symmetric 1 -window-size 4 -vocab-file {} -memory 8.0 -distance-weighting 0 < {} > {}'.format(vocab,corpus,cooccur_matrix))


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

def prepare_matrix_for_pmi(data,num_words,smooth=0,cds=True):
	mat = np.zeros((num_words,num_words))
	for x,i in enumerate(data):
		mat[i[1]-1][i[0]-1] = i[2]
		mat[i[0]-1][i[1]-1] = i[2]
	old_mat = coo_matrix(mat).tocsr()    
	#save_npz('mensrights_coo_matrix.npz',coo_mat)
	#old_mat = load_npz(count_path).tocsr()

	smooth = old_mat.sum() * smooth
	row_probs = old_mat.sum(1) + smooth
	col_probs = old_mat.sum(0) + smooth
	if cds:
		col_probs = np.power(col_probs, 0.75)
	row_probs = row_probs / row_probs.sum()
	col_probs = col_probs / col_probs.sum()
	print('Prepared original matrix for PPMI')
	return old_mat,row_probs,col_probs,smooth





def get_ppmi_svd(subreddit,vocab_dir,cooccur_dir,ppmi_svd_dir,smooth=0,cds=True,neg=1,normalize=False):
	vocab_file = os.path.join(vocab_dir,subreddit + '.txt')
	with open(vocab_file) as f:
		vocab = [v.split('\n')[0].split(' ')[0] for v in f.readlines()]

	infile = os.path.join(cooccur_dir,subreddit + '.bin')
	dt = np.dtype([('word1',np.uint32),('word2',np.uint32),('count',np.double)])
	data = np.fromfile(infile, dtype=dt) 
	old_mat,row_probs,col_probs,smooth = prepare_matrix_for_pmi(data,len(vocab),smooth,cds)
	ppmi_mat = make_ppmi_mat(old_mat, row_probs, col_probs,smooth,neg,normalize)
	# save_npz(out_path,ppmi_mat)
	# ppmi_mat = load_npz(out_path)
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



def run_sentprop(subreddit,ppmi_svd_dir,socialsent_lexicons_dir,vocab_dir):
	program = 'python make_sent_lexicons.py ' + subreddit + " " + ppmi_svd_dir + " " + socialsent_lexicons_dir + " " + vocab_dir
	os.system(program)

def make_sent_lexicon(subreddit):
	vocab_dir = 'vocab_counts'
	concat_dir = 'concat_subs'
	cooccur_dir = 'cooccur_matrices_unweighted'
	ppmi_svd_dir = 'ppmi_svd_vectors'
	socialsent_lexicons_dir = 'socialsent_lexicons_ppmi_svd_top5000_boots'
	socialsent_vectors_dir = 'socialsent_vectors_ppmi_svd_boots'
	# get_vocabulary(subreddit,concat_dir,vocab_dir)
	# get_count_matrices(subreddit,concat_dir,vocab_dir,cooccur_dir)
	# get_ppmi_svd(subreddit,vocab_dir,cooccur_dir,ppmi_svd_dir)
	run_sentprop(subreddit,ppmi_svd_dir,socialsent_lexicons_dir,vocab_dir)


""" 
Making the sentprop vectors requires all the lexicons so it must wait
It involves combining the vocabulary, normalizing (look back at the paper) and saving the vector
"""

def make_sentprop_vectors(subreddit,socialsent_lexicons_dir,socialsent_vectors_dir):
	return 0 

def main():
	#subreddit_list = ['femalefashionadvice']
	subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes','london','actuallesbians','askmen','askwomen','askgaybros','xxfitness']
	for subreddit in subreddit_list:
		print(subreddit)
		make_sent_lexicon(subreddit)

if __name__ == "__main__":
	main()