import os 
import glob 


concat_docs = glob.glob('concat_subs/*.txt')
subreddit_names = [os.path.basename(d)[:-4] for d in concat_docs] #removes the .txt
for subreddit in subreddit_names:
	corpus = os.path.join('concat_subs',subreddit + '.txt')
	vocab = os.path.join('vocab_counts',subreddit + '.txt')
	cooccur_matrix = os.path.join('cooccur_matrices',subreddit + '.bin')
	cooccur_shuffled = os.path.join('cooccur_shuffled',subreddit + '.bin')
	glove_vector = os.path.join('glove_vectors',subreddit + '.txt')
	gradsq_file = os.path.join('glove_vectors',subreddit + '_gradsq')

	os.system('./vocab_count -verbose 2 -max-vocab 100000 -min-count 5 < {} > {}'.format(corpus,vocab))
	os.system('./cooccur -verbose 2 -symmetric 1 -window-size 5 -vocab-file {} -memory 8.0 < {} > {}'.format(vocab,corpus,cooccur_matrix))
	os.system('./shuffle -verbose 2 -memory 8.0 < {} > {}'.format(cooccur_matrix,cooccur_shuffled))
	os.system('./glove -input-file {} -vocab-file {} -save-file {} -gradsq-file {} -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2 -iter 15'.format(cooccur_shuffled,vocab,glove_vector, gradsq_file))