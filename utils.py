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
import string
from collections import OrderedDict
from os import listdir
from os.path import isfile, join

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
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    
    for file in onlyfiles:
        filepath = os.path.join(str(PATH), file)
        with open(filepath, 'r') as f:
            speech = f.read().split("\n\n")
            para = [line.replace('\n', ' ') for line in speech]
            year = re.findall('(?<=_).*?(?=\.)', file)[0]
            president = re.findall('([^/.]+)(?=_)', file)[0]
            speeches[(president, year)] = para

    return speeches

def contains_multiple_words(s):   
    """
    function determining if the string is more than 1 word
    
    input:
        s: a string
    output:
        True if there are more than 1 element after applying a split function. False else.
    """
    if len(s.split()) > 1:
        return True
    else:
        return False

def clean_words(speech_dict):
    """
    Creates a dictionary which keys are the
    tuple (president,year), and its value a list of
    tokenized words without StopWords.
    
    Inputs: Dictionary with speeches
    Returns: Dictionary
    """

    clean_dict = {}
    tokenizer = RegexpTokenizer(r'\w+') # we are not using it
    stopWords = set(stopwords.words('english'))
    for k, v in speech_dict.items():
        if k not in clean_dict:
            clean_dict[k]= []
        for line in v:
            line = str(line).lower()
            line = line.lower().replace('"',"")
            line = tokenizer.tokenize(line)
            line = ' '.join(line)
            if contains_multiple_words(line) or line not in stopWords:
                clean_dict[k].append([line])
                
    clean_dict = {k: [val for sublist in v for val in sublist] for k,v in clean_dict.items()}
    return clean_dict

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
    """
    combines counts of current and previous paragraphs
    
    inputs:
        count_dict: the dictionary of counted noun phrases from the current paragraph/speech
        use_dict: the master dictionary that's tracking counts of all noun phrases across the entire corpus
    outputs:
        a dictionary with counts from count_dict and use_dict combined
    """
    
    for word in count_dict.keys():
        if word in list(use_dict.keys()):
            use_dict[word] += count_dict[word]
        else:
            use_dict[word] = count_dict[word]
    
    return use_dict

def top_x(data, x):
    """
    turns the noun phrase data into counts of noun phrases, then finds the top 1000 most stated noun phrases for the co-occurrence matrix.
    
    inputs:
        data: speeches that are cleaned and preprocessed into noun phrase chunks
        x: the number of top entires we want
    outputs:
        dict_use: total count dictionary
        sorted: the top x most referenced noun phrases across all of the speeches.
    """

    dict_use = {}
    counter = 1
    for n, v in enumerate(data.items()):
        
        name = v[0]
        paragraph = v[1]
    
        counts = dict((str(x),paragraph.count(x)) for x in set(paragraph))
    
        dict_use = combine_counts(counts, dict_use)
    
        counter += 1
        
    return dict_use, sorted(dict_use, key=dict_use.get, reverse=True)[:x]

def co_occurence_matrix(dictionary_of_speeches, dataframe=False, csv=False):
    
    '''Creates a co-occurence matrix with the noun-phrases
    that appear in each paragraph of each speech. 
    Inputs (dictionary): Dictionary of speeches
    dataframe (boolean): if true, returns a dataframe
    csv (boolean) : if true, writes a csv file with the name "co_occurence.csv" 
    '''
    list_of_paragraphs=[]
    for speech in list(dictionary_of_speeches.values()):
        for paragraph in speech:
            list_of_paragraphs.append(paragraph)
    documents=[]
    for paragraph in list_of_paragraphs:
        nouns = spacy_fxn_ls (paragraph)
        list_of_nouns=[]    
        for noun in nouns:
            list_of_nouns.append(str(noun))
        documents.append(list_of_nouns)
    
    noun_set=[]
    for ls in documents:
        for noun in ls:
            if noun not in noun_set:
                noun_set.append(noun)
    # OrderedDict to count each occurence in each paragraph
    occurrences = OrderedDict(((noun), OrderedDict(((noun), 0) for noun in noun_set)) for noun in (noun_set))

    # Find the co-occurrences:
    for l in documents:
        for i in range(len(l)):
            for item in l[:i] + l[i + 1:]:
                occurrences[l[i]][item] += 1
    rows = []
    columns=[]
    for noun, values in occurrences.items():
        #print(name, ' '.join(str(i) for i in values.values()))
        columns.append(noun)
        rows.append(values.values())
    if dataframe:
        df = pd.DataFrame(list(rows), columns=columns, index=columns) 
    if csv:
        df.to_csv("co_occurence.csv", sep = ",")   
    return df

def spacy_fxn_ls(strings):
    """Returns a list of noun phrases
    Inputs: string
    Returns: list"""
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(strings)
    l = []
    for token in doc.noun_chunks:
        l.append(token)
    print(l)
    return l

if __name__ == '__main__':
    go()
