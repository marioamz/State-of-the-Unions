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
                if word not in stopWords:
                    if word in dict_use[key]:
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
            for word in tokenizer.tokenize(l.lower()):
                if word not in stopWords:
                    if word in dict_use[k].keys():
                        #rint(dict_use[k].keys())
                        dict_use[k][word] += 1
                    else:
                        dict_use[k][word] = 1  
    return dict_use

def counts_overall(data):
    '''
    Function to create a dictionary of words counts by president and year. A larger function that allows the user to choose the data cut calls this function.
    
    inputs: 
        data: speech text from reading_data function above
        keys: the keys of the dictionary outputted in reading_data function above
        dict_use: the dictionary that is to be filled
    output:
       dict_use: a dictionary of dictionary of counts.
    '''
    dict_use = {}
    for n, key in enumerate(data):
        if n == 0:
            dict_use = data[key]
        else:
            for word in data[key].keys():
                if word in dict_use.keys():
                    dict_use[word] += data[key][word]
                else:
                    dict_use[word] = data[key][word]  
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
        return counts_by_pres(data, keys, dict_use)
    elif breakout == "by pres year":
        return counts_by_pres_year(data, keys, dict_use)
    elif breakout == "overall":
        a = counts_by_pres_year(data, keys, dict_use)
        return counts_overall(a)
    else:
        raise Exception('breakout should either be "by pres", "by pres year", or "overall". The value of breakout was: {}'.format(breakout))
        

def plot_top_words(data, list_of_presidents, top_words,  PATH, vertical=True, create_pngs = False):
    
    '''    
    Creates a barplot for the top words for each president
    Inputs:
        data: dictionary with wordcounts by president
        list_of_presidents (list of strings): if ["all"], it creates a plot for all presidents in the
        data. Otherwise, it needs a list with the names of interest
        top_words (int): number of k-top words
        vertical: if True, it plots vertical bars; else, bars are horizontally aligned
    Returns:
            It does not return anything but shows the plot.
    '''    
    
    if list_of_presidents[0] == "all":
        for president in data.keys():
            d = data[president]
            top_list = sorted(d, key=d.get, reverse=True)[:top_words]
            top_five = {word:val for word,val in d.items() if word in top_list}
            dataframe= pd.DataFrame(top_five.items(), columns=['Words', 'Count'])
            if vertical:
                plot = sns.barplot(x="Words", y="Count", data=dataframe, order = top_list).set_title(president)
                plt.show()
                plot = plot.get_figure()
                #if create_pngs == True: # Change the folder/path where you want to save the plots.
                #    plot.savefig(PATH + president + ".png")
            else:
                plot = sns.barplot(y="Words", x="Count", data=dataframe, order = top_list).set_title(president)
                plt.show()
                plot = plot.get_figure()
                #if create_pngs == True: # Change the folder/path where you want to save the plots
                #    plot.savefig(PATH + president + ".png")

    else:
        for president in list_of_presidents:
            d = data[president]
            top_list = sorted(d, key=d.get, reverse=True)[:5]
            top_five = {word:val for word,val in d.items() if word in top_list}
            dataframe= pd.DataFrame(top_five.items(), columns=['Words', 'Count'])
            if vertical:
                plot = sns.barplot(x="Words", y="Count", data=dataframe, order = top_list).set_title(president)
                plt.show()
                plot = plot.get_figure()
                #if create_pngs == True: # Change the folder/path where you want to save the plots
                #    plot.savefig(PATH + president + ".png")

            else:
                plot = sns.barplot(y="Words", x="Count", data=dataframe, order = top_list).set_title(president)
                plt.show()
                plot = plot.get_figure()
                #if create_pngs == True: # Change the folder/path where you want to save the plots
                #    plot.savefig(PATH + president + ".png")

if __name__ == '__main__':
    go()
