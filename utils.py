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
import numpy as np
import community
import glob
import os
import re
import sys
import spacy
import string
from operator import itemgetter
from os import listdir
from os.path import isfile, join
import textdistance
from itertools import groupby, chain
from collections import OrderedDict
import sklearn
from sklearn import cluster
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

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


def convert_dict_to_list(dictionary):
    """ it creates a list of lists in which each list is a paragraph
    coming from a speech
    Inputs: Dictionary of tuples (President, year): list of tuples (word, pagraph number )
    Returns: List of lists"""

    list_document=[]

    for key, value in dictionary.items():
        #print(key)
        if len(value)>0:
            for i in range((max(value,key = itemgetter(1))[1]) +1):
                ls=[]
                for noun in value:
                    if noun[1]==i:
                        ls.append(noun[0])
                if len(ls)>0:
                    list_document.append(ls)

    return list_document


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


def co_oc_matrix(documents, dataframe=False, csv=False):

    '''Creates a co-occurence matrix with the noun-phrases
    that appear in each paragraph of each speech.
    Inputs (list of lists): list of noun phrases (each paragraph is a list)
    dataframe (boolean): if true, returns a dataframe
    csv (boolean) : if true, writes a csv file with the name "co_occurence.csv"
    '''

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
        for column in columns:
            df.at[column, column] = 1
    if csv:
        df.to_csv("co_occurence.csv", sep = ",")

    return df


def spacy_fxn_ls(strings):
    """Returns a list of noun phrases
    Inputs: string
    Returns: list
    """

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(strings)

    l = []
    for token in doc.noun_chunks:
        l.append(token)
    print(l)

    return l


def pairwise_similarity(df, method):
    '''
    This function takes in the path name for an excel file and
    returns a co-occurrence matrix with pairwise values.

    It does the work based on one of two methods:
        - paper method
        - cosine similarity
    '''

    if method == 'paper':

        total_pars = 21143
        for i in df.columns:
            colsum = df[i].sum()
            for j, r in df.iterrows():
                cosum = df[j].sum()
                a = np.log((r/total_pars) / (colsum/total_pars)*(cosum/total_pars)+1)
                df[i] = a

        return df

    else:

        co_matrix = pd.DataFrame(cosine_similarity(df), index=df.columns, columns = df.columns)

        return co_matrix


def network_graph(df, method):
    '''
    This graph takes the pairwise co-occurrence matrix and returns
    a network where the nodes are noun phrases and the edges are their
    similarity to each other.

    It also returns a list of clusters, with the words that made it up.
    '''

    clusters = []
    # establish graph
    graph = nx.Graph(df)

    edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())

    if method == 'community':
         #first compute the best partition
         partition = community.best_partition(graph)

    else:
        # first calculate k-means unsupervised
        kmeans = cluster.KMeans(n_clusters = 8).fit(df)
        df['scores'] = kmeans.labels_
        partition = df['scores'].to_dict()

    #drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(graph)
    count = 0.

    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                            if partition[nodes] == com]
        clusters.append(list_nodes)
        nx.draw_networkx_nodes(graph, pos, list_nodes, node_size = 5,
                                            node_color = str(count / size))

    nx.draw_networkx_edges(graph, pos, alpha=0.2, edge_color = weights)
    #nx.draw_networkx_labels(graph, pos, font_size = 4, alpha = 0.6)
    #plt.figure(figsize=(100,100))
    plt.savefig("testgraph.png", dpi=500)

    return clusters


if __name__ == '__main__':
    go()
