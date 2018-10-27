# Using sentiment induction to understand variation in gendered online communities
## Setup
This directory is built on top of [SocialSent](https://github.com/williamleif/socialsent/tree/master/socialsent). To run several of the code files, you should first download this `socialsent` folder and place it inside the `code` folder. 
## Data
We used Reddit comments between May 2016 and April 2017 from nine gendered communities that are within the most popular 400 subreddits: r/actuallesbians, r/askgaybros, r/mensrights, r/askmen, r/askwomen, r/xxfitness, r/femalefashionadvice, r/malefashionadvice, and r/trollxchromosomes. We used a dataset provided by the Stanford Infolab, but Reddit comment data is also available publicly in various forms: on BigQuery [here](https://pushshift.io/using-bigquery-with-reddit-data/) or via [download](https://files.pushshift.io/reddit/) with an API [here](https://github.com/pushshift/api). 
## Code
- `clustering.py` contains code for clustering user and text representations of subreddits. 
- `create_docs.py` concatenates reddit comments into large documents, one per subreddit
- `create_subreddit_list.py` shows how we narrowed down to our target subreddits
- `misalignment.py` examines differences between text and user representations
- `pipeline.py` creates sentiment lexicons with SentProp
- `plot_sim_correlations.ipynb` contains analysis and plots of sentiment 
- `subreddit_counts.py` calculates basic statistics about our data
- `variance_sentiment.py` finds words with high variance in sentiment across subreddits
## Lexicons
The induced sentiment lexicons we analyzed in our paper can be found [here](https://github.com/lucy3/reddit-sent/tree/master/logs/socialsent_lexicons_ppmi_svd_top5000). 
