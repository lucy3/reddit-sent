"""
From set(listofsubreddits.txt), 
we subtract set(defaults.txt). 

We also look at token counts for
various subreddits to find a good
window of time. 
"""
import json
from collections import Counter
import numpy as np

SR_LIST = '../data/listofsubreddits.txt'
DEFAULTS = '../data/defaults.txt'
NO_DEFAULTS = '../data/subreddits_no_defaults.txt'
TOK_COUNTS = '../logs/token_counts.json'

def subtract_defaults(): 
    reddits = set()
    with open(SR_LIST, 'r') as inputfile: 
        for line in inputfile: 
            if line.startswith('/r/'): 
                reddits.add(line.strip())
    defaults = set()
    with open(DEFAULTS, 'r') as inputfile: 
        for line in inputfile: 
            if line.startswith('/r/'): 
                line = line.strip()
                if line.endswith('/'): 
                    line = line[:-1]
                defaults.add(line)
    reddits = reddits - defaults
    with open(NO_DEFAULTS, 'w') as outputfile: 
        for sub in reddits: 
            outputfile.write(sub.lower() + '\n')

def inspect_counts(): 
    reddits = set()
    with open(NO_DEFAULTS, 'r') as inputfile: 
        for line in inputfile: 
            reddits.add(line.strip()[3:])
    gender_related = set(['MaleFashionAdvice',
                'everymanshouldknow',
                'askmen',
                'frugalmalefashion',
                'MensRights',
                'malelifestyle',
                'trollychromosome',
                'malelivingspace',
                'malehairadvice',
                'malefashion',
                'TwoXChromosomes',
                'askwomen',
                'LadyBoners',
                'TrollXChromosomes',
                'femalefashionadvice',
                'xxfitness',
                'TheGirlSurvivalGuide',
                'abrathatfits',
                'badwomensanatomy',
                'malelifestyle',
                'malelivingspace',
                'TheGirlSurvivalGuide',
                'MensRights',
                'feminism'])
    with open(TOK_COUNTS, 'r') as inputfile: 
        data = json.load(inputfile)
    print reddits - set(data.keys())
    top_sr = set()
    for sr in data: 
        if data[sr] > 10**7: 
            top_sr.add(sr)
    print len(top_sr)
    print gender_related & top_sr

def main(): 
    subtract_defaults()
    inspect_counts()

if __name__ == '__main__':
    main()