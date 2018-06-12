import pickle 
import numpy as np
import pandas as pd 
import os
import csv

 
def create_dataframe(vector_dir,subreddit_list):
	vecs = {}
	for sub in subreddit_list:
		vec_file = os.path.join(vector_dir,sub + '.npy')
		vec = np.load(vec_file)
		vecs[sub] = vec
	df = pd.DataFrame(vecs)
	return df

def get_variances(df):
	variances = df.var(axis=1)
	idx = list(np.argsort(variances, axis=1))
	return idx,variances


def main():
	vector_dir = 'socialsent_vectors_ppmi_svd_top5000'
	subreddit_list = ['femalefashionadvice','malefashionadvice','mensrights','trollxchromosomes','london','actuallesbians','askmen','askwomen','askgaybros','xxfitness']
	vocab_file = 'vocab_socialsent_ordered.pkl'
	variance_file = 'variance_socialsent.tsv'
	df = create_dataframe(vector_dir,subreddit_list)
	idx,variances = get_variances(df)
	for i in idx[-30:][::-1]:
		print i,vocab_list[i],df[i]

	# with open(vocab_file,'r') as f:
	# 	vocab_list = pickle.load(f)

	# with open(variance_file,'w') as tsvin:
	# 	tsvin = csv.writer(tsvin,delimiter='\t')
	# 	for i in idx:
	# 		tsvin.writerow((vocab_list[i],variances[i]))
		


if __name__ == "__main__":
	main()
