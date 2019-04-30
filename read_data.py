# Script for reading in the data

import read_data as rd
import importlib
importlib.reload(rd)
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

def counts_by_pres(data, keys, dict_use):
    '''
    Function to create a dictionary of words counts by president. A larger function that allows the user to choose the data cut calls this function.
    
    inputs: 
        data: speech text from reading_data function above
        keys: the keys of the dictionary outputted in reading_data function above
        dict_use: the dictionary that is to be filled
    output:
       dict_use: a dictionary of dictionary of counts.
    '''
    for k in keys:
        key = k[0]
        for l in data[k]:
            for word in tokenizer.tokenize(l.lower()):
                if word in dict_use[key] and word not in stopWords:
                    dict_use[key][word] += 1
                else:
                    dict_use[key][word] = 1
    return dict_use

def counts_by_pres_year(data, keys, dict_use):
    '''
    Function to create a dictionary of words counts by president and year. A larger function that allows the user to choose the data cut calls this function.
    
    inputs: 
        data: speech text from reading_data function above
        keys: the keys of the dictionary outputted in reading_data function above
        dict_use: the dictionary that is to be filled
    output:
       dict_use: a dictionary of dictionary of counts.
    '''
    for k in keys:
        for l in data[k]:
            for word in l.split():
                if word in dict_use[k]:
                    dict_use[k][word] += 1
                else:
                    dict_use[k][word] = 1
    return dict_use

def make_dict(data, keys, dict_use, breakout):
    '''
    Function that calls other functions based on the breakout specified.
    
    inputs: 
        data: speech text from reading_data function above
        keys: the keys of the dictionary outputted in reading_data function above
        dict_use: the dictionary that is to be filled
        breakout: user defined entry that allows them to choose the type of breakout wanted.
    output:
       dict_use: a dictionary of dictionary of counts.
    '''
    if breakout == "by pres":
        print("by pres")
        return counts_by_pres(data, keys, dict_use)
    if breakout == "by pres year":
        print("by pres year")
        return counts_by_pres_year(data, keys, dict_use)
    else:
        raise Exception('breakout should either be "by pres" or "by pres year". The value of breakout was: {}'.format(breakout))

if __name__ == '__main__':
    go()
