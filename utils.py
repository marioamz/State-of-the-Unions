# Script for reading in the data

import utils as ut
import importlib
importlib.reload(ut)
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os
import re
import sys
import spacy
import string

from os import listdir
from os.path import isfile, join

import textdistance
from itertools import groupby, chain

stopWords = set(stopwords.words('english'))
stopWords.add("000")
tokenizer = RegexpTokenizer(r'\w+')

lemmatizer = WordNetLemmatizer()

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

def chunks(dictionary):
    '''
    This function takes in a dictionary of speeches and creates
    noun phrase observations for each.
    '''

    new_dict = {}

    nlp = spacy.load('en_core_web_sm')

    for m, (k, v) in enumerate(dictionary.items()):
        noun_phrase = []
        for n, paragraph in enumerate(v):
            doc = nlp(paragraph)
            for token in doc.noun_chunks:
                noun_phrase.append((token, n))
        new_dict[k] = noun_phrase
        
        #print(m)
        #if m == 20:
            
        #    break
    return new_dict

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

def nltk2wn_tag(nltk_tag):
    """
    get POS tagging to be able to better lemmatize words
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def lemmed(token):
    """
    lemmatize all words
    """
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(token))  
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    
    for word, tag in wn_tagged:
        if tag is None:            
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
            
    return " ".join(res_words)

def clean_words(speech_dict):
    """
    Creates a dictionary which keys are the
    tuple (president,year), and its value a list of
    tokenized words without StopWords.
    
    Inputs: Dictionary with speeches
    Returns: Dictionary
    """
    clean_dict = {}
    tokenizer = RegexpTokenizer(r'\w+')
    stopWords = set(stopwords.words('english'))
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for n, (k, v) in enumerate(speech_dict.items()):
        if k not in clean_dict:
            clean_dict[k] = []
        for line, para_num in v:
            line = str(line).lower().strip()
            line = ''.join(filter(whitelist.__contains__, line))

            if contains_multiple_words(line) or line not in stopWords:
                lem_word = lemmed(line)
                clean_dict[k].append([(lem_word, para_num)])
        
    clean_dict = {k: [val for sublist in v for val in sublist] for k,v in clean_dict.items()}
    return clean_dict

def count_words(data):
    """
    turns the noun phrase data into counts of noun phrases, then finds the top 1000 most stated noun phrases for the co-occurrence matrix.
    
    inputs:
        data: speeches that are cleaned and preprocessed into noun phrase chunks
        x: the number of top entires we want
    outputs:
        dict_use: total count dictionary
        sorted: the top x most referenced noun phrases across all of the speeches.
    """
    list_of_words = []
    for n, x in enumerate(data.values()):
        first_words = [i[0] for i in x]
        list_of_words = list_of_words + first_words
        #print(n)
    #print(list_of_words)
    counts = Counter(list_of_words)
    return counts   

def top_x(dict_use, x):
    """
    sort and take only top 1000 words/noun phrases
    """
    return sorted(dict_use, key=dict_use.get, reverse=True)[:x]
    #print(words)
    
    #return sorted(dict_use, key=dict_use.get, reverse=True)[:x]

def limit(full_data, top_words_data):
    """
    limit the noun phrases by speech and paragraph down to top 1000 words/noun phrases only
    """
    for n, x in enumerate(full_data.keys()):
        new_list = [item for item in full_data[x] if item[0] in top_words_data]
        full_data[x] = new_list 
    
    return full_data

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
