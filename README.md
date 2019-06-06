# State-of-the-Unions
An NLP analysis of State of the Union speeches from 1789 to 2018. 

## In this repository there are x files:

utils_preprocessing.py
- Author: Alix
- Purpose: houses all of the functions necessary for preprocessing, such as creating noun phrases, lemmatizing noun phrases, mapping noun phrases, and periodization 

pipeline_preprocessing.ipynb
- Author: Alix & Mario
- Purpose: Calls all of the preprocessing functions in the utils_preprocessing.py

Chunking.ipynb
- Author: Mario
- Purpose: ___

Topic_Model.ipynb
- Author:
- Purpose:

pipeline.ipynb
- Author:
- Purpose:

utils.py
- Author: 
- Purpose:

pipeline_preprocessing_regex.ipynb
- Author: Alix
- Purpose: preprocessing pipeline, only for regex

pipeline_preprocessing_spacy.ipynb
- Author: Alix
- Purpose: preprocessing pipeline, only for spacy


## There are 3 additional folders: 

EDA
- Purpose: exploratory data analysis and notebooks for the mid-quarter presentation

LDA_viz
- Purpose:

Networks:
- Purpose:

## To run this code you need the following libraries:

- importlib
- nltk
- collections import Counter
- nltk.tokenize import sent_tokenize, word_tokenize
- nltk.corpus import stopwords, wordnet
- nltk.tokenize import RegexpTokenizer
- nltk.stem.wordnet import WordNetLemmatizer
- matplotlib.pyplot as plt
- seaborn as sns
- pandas as pd
- glob
- os
- os import listdir
- os.path import isfile, join
- re
- sys
- spacy
- string
- numpy as np
- pandas as pd
- math
- sklearn.metrics.pairwise import cosine_similarity
- sklearn import preprocessing
- min_max_scaler = preprocessing.MinMaxScaler()
- textdistance
- itertools
- itertools import groupby, chain
- community
- numpy as np
- networkx as nx
- sklearn import cluster
- operator import itemgetter
- collections import OrderedDict
- gensim
- gensim.corpora as corpora
- gensim.utils import simple_preprocess
- gensim.models import CoherenceModel
- logging
- warnings

For plotting LDA:
- pyLDAvis
- pyLDAvis.gensim

