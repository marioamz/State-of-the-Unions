# Script for reading in the data

import utils as ut
import importlib
importlib.reload(ut)
import nltk
import collections
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os
import re
import sys
import spacy

stopWords = set(stopwords.words('english'))
stopWords.add("000")
tokenizer = RegexpTokenizer(r'\w+')

def go():
    '''
    go function that runs the script.
    '''

    data = reading_data(str(sys.argv[1]), str(sys.argv[2]))
    print(data)
    return data


def reading_data(PATH, filetype):
    '''
    This function reads in the corpus of the speeches and turns
    them into dictionaries of the form:
        - Key: (President, year), Value: list of paragraphs in speech

    It takes in the PATH where the text files are located, and
    the type of the files (in this case, '*.txt')
    '''

    speeches = {}

    filepaths = glob.glob(os.path.join(str(PATH), str(filetype)))

    for file in filepaths:
        with open(file, 'r') as f:
            speech = f.read().split("\n\n")
            para = [line.replace('\n', ' ') for line in speech]
            year = re.findall('(?<=_).*?(?=\.)', file)[0]
            president = re.findall('([^/.]+)(?=_)', file)[0]
            speeches[(president, year)] = para

    return speeches

def chunks(dictionary):
    '''
    This function takes in a dictionary of speeches and creates
    noun phrase observations for each.
    '''

    new_dict = {}

    nlp = spacy.load('en_core_web_sm')

    for k, v in dictionary.items():
        noun_phrase = []
        for paragraph in v:
            doc = nlp(paragraph)
            for token in doc.noun_chunks:
                noun_phrase.append(token)
        new_dict[k] = noun_phrase
        
    return new_dict

def combine_counts(count_dict, use_dict):
    for word in count_dict.keys():
            if word in list(use_dict.keys()):
                use_dict[word] += count_dict[word]
                
            else:
                use_dict[word] = count_dict[word]
    
    return use_dict

def top_x(data, x):

    dict_use = {}
    counter = 1
    for n, v in enumerate(data.items()):
        print(counter)
        name = v[0]
        paragraph = v[1]
    
        counts = dict((str(x),paragraph.count(x)) for x in set(paragraph))
    
        dict_use = combine_counts(counts, dict_use)
    
        counter += 1
        
    return dict_use, sorted(dict_use, key=dict_use.get, reverse=True)[:x]

if __name__ == '__main__':
    go()
