# Script for reading in the data

import utils_preprocessing as up
import importlib
importlib.reload(up)
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
import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

from os import listdir
from os.path import isfile, join

import textdistance
import itertools
from itertools import groupby

stopWords = set(stopwords.words('english'))
stopWords.add("000")
tokenizer = RegexpTokenizer(r'\w+')

grammar = "NP: {<DT>?<JJ.*><NN.*>+}"
cp = nltk.RegexpParser(grammar)

lemmatizer = WordNetLemmatizer()


def go():
    '''
    go function that runs the script.
    '''

    data = reading_data(str(sys.argv[1]), str(sys.argv[2]))
    #print(data)
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
            if year != "1790":
                speeches[(president, year)] = para

    su = 0
    for i, x in speeches.items():
        su = su + len(x)

    return speeches, su

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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def use_regex(paragraph, num):
 
    sentences = nltk.sent_tokenize(paragraph)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    noun_phrase = []
    
    for sentence in sentences:
        result = cp.parse(sentence)
        for word in result:
            if type(word) == tuple:
                np = word[0].lower().translate(str.maketrans('', '', string.punctuation))
                if np in ['l', 'm', 'p', 'u', 'v']:
                    pass
                elif (nltk2wn_tag(word[1]) is not None) and (np not in stopWords) and (not is_number(np)) and (len(np)>1):
                    noun_phrase.append((lemmatizer.lemmatize(np, nltk2wn_tag(word[1])), num))
                elif (np not in stopWords) and (not is_number(np)) and (len(np)>1):
                    noun_phrase.append((lemmatizer.lemmatize(np), num))
                    
            else:
                np_list = []
                for w in word:
                    w1 = w[0].lower().translate(str.maketrans('', '', string.punctuation))
                    if nltk2wn_tag(w[1]) == None:
                        np_list.append(lemmatizer.lemmatize(w1))
                        
                    else:
                        np_list.append(lemmatizer.lemmatize(w1, nltk2wn_tag(w[1])))
                
                if len(np_list) > 0:
                    noun_phrase.append((" ".join(np_list), num))
                    
    return noun_phrase
                    
    
    
def chunks(dictionary, noun_phrase_type):
    '''
    This function takes in a dictionary of speeches and creates
    noun phrase observations for each.
    '''

    new_dict = {}

    for m, (k, v) in enumerate(dictionary.items()):
        
        if noun_phrase_type == "spacy":
            noun_phrase = []
            nlp = spacy.load('en_core_web_sm')
            
            for n, paragraph in enumerate(v):
                doc = nlp(paragraph)
                
                for token in doc.noun_chunks:
                    noun_phrase.append((token, n))
                    
            new_dict[k] = noun_phrase

        elif noun_phrase_type == "regex":
            for n, paragraph in enumerate(v):
                if  n == 0:
                    new_dict[k] = use_regex(paragraph, n)
                else:
                    new_dict[k] += use_regex(paragraph, n)
                    #print(new_dict)
            
        else:
            print("pleas use 'spacy' or 'regex'")
    
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


def jaccard(a, b):
    '''
    This function takes the jaccard similarity between two
    noun phrases in order to return their distance to each
    other.
    '''

    a_set = set(a.split())
    b_set = set(b.split())
    c = a_set.intersection(b_set)
    if (len(a_set) + len(b_set) - len(c)) > 0:
        return float(len(c)) / (len(a_set) + len(b_set) - len(c))


def first_calc(data):
    """
    1. calculate linear jaccard similarity
    """

    comparison = "america is wonderfully weird"
    diff_calcs = []
    for word in data:
            calc = textdistance.jaccard.normalized_distance(word, comparison)
            diff_calcs.append((word, calc))

    return sorted(diff_calcs, key=lambda tup: tup[1])


def split_list(alist, wanted_parts=1):
    """
    2. split the massive list into smaller lists
    """

    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def within_calcs(data, thresh):
    """
    3. calculate jaccard similarity within each of the smaller lists and classify - THIS IS WHAT NEEDS ATTENTION.
    """

    changed_words = {}

    for word_n, word in enumerate(data):
        if len(word) > 0:
            for compare_word_n, compare_word in enumerate(data):
                if len(compare_word) > 0:
                    calc = jaccard(word, compare_word)
                    #print(calc)
                    if calc is not None:

                        if (calc > thresh) & (word != compare_word) & (word in changed_words.keys()):
                            changed_words[word].append(compare_word)

                        elif (calc > thresh) & (word != compare_word) & (word in [b for a in changed_words.values() for b in a]):
                            k = [key for key, value in changed_words.items() if word in value]
                            changed_words[k[0]].append(compare_word)

                        elif (calc > thresh) & (word != compare_word) & (compare_word in changed_words.keys()):
                            changed_words[compare_word].append(word)

                        elif (calc > thresh) & (word != compare_word) & (compare_word in [b for a in changed_words.values() for b in a]):
                            k = [key for key, value in changed_words.items() if compare_word in value]
                            changed_words[k[0]].append(word)

                        elif (calc > thresh) & (word != compare_word):
                            changed_words[word] = [compare_word]

    return changed_words


def word_changes(data, thresh, num_lists):
    """
    1. calculate linear jaccard similarity
    2. split the massive list into smaller lists
    3. calculate jaccard similarity within each of the smaller lists and classify - THIS IS WHAT NEEDS ATTENTION.
    """

    flat_list = set(item[0] for sublist in data.values() for item in sublist)
    diff_calcs = first_calc(flat_list)
    lists_split = split_list(diff_calcs, num_lists)
    result = {}
    for n, l in enumerate(lists_split):
        l_1 = [item[0] for item in l]
        second_diff_calcs = within_calcs(l_1, thresh)
        result = {**result, **second_diff_calcs}

    return result


def lemmed_phrases(changed_data, clean_data):
    '''
    This function takes in phrases and lems the words in
    it.
    '''

    flat_list = set(item for sublist in changed_data.values() for item in sublist)

    for n, entry in enumerate(clean_data):
        words_list = [item[0] for item in clean_data[entry]]
        for word in words_list:
            if word in flat_list:
                k = [key for key, value in changed_data.items() if word in value]
                clean_data[entry] = [(k[0], item[1]) if item[0]==word else item for item in clean_data[entry]]
    return clean_data


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
    for n, x in enumerate(data):
        d = data[x]
        first_words = [i[0] for i in d]
        list_of_words = list_of_words + first_words

    counts = Counter(list_of_words)
    return counts


def top_x(dict_use, x):
    """
    sort and take only top 1000 words/noun phrases
    """

    return sorted(dict_use, key=dict_use.get, reverse=True)[:x]


def limit(full_data, top_words_data):
    """
    limit the noun phrases by speech and paragraph down to top 1000 words/noun phrases only
    """

    for n, x in enumerate(full_data.keys()):
        new_list = [item for item in full_data[x] if item[0] in top_words_data]
        full_data[x] = new_list

    return full_data

def corpus_tfidf(limited_data, counted_data, top_data):
    """
    calculates tfidfs across the corpus
    """

    index = top_data
    columns = limited_data.keys()
    df = pd.DataFrame(index=index, columns=columns)
    df = df.fillna(0)

    for n, x  in enumerate(limited_data):
        l = [item[0] for item in limited_data[x]]
        counts = Counter(l)
        c = dict(counts)
        #print(c)
        df1 = pd.DataFrame.from_dict(c, orient='index')
        df1.rename(columns={0:x}, inplace=True)
        df.update(df1)
        for m, (index, row) in enumerate(df.iterrows()):
            if counted_data[index] > 0:
                df[x][index] = df[x][index]*(math.log(len(columns)/counted_data[index]))
            else:
                df[x][index] = 0


        df[x] = (df[x]/df[x].sum(0))

    return df.T


def calc_sum(full_data, years_data):
    """
    calculates sums across combos of years
    """

    combo_data = list(itertools.combinations(years_data, 2))

    total = 0
    for n, item in enumerate(combo_data):
        a = full_data.at[item[0], item[1]]
        b = a/(len(years_data)*len(years_data)-1)
        total += b

    return total



def periodization(tfidf_data):
    """
    calculates stuff from the paper to find distinct periods
    """
    
    tfidf_data.index = tfidf_data.index.droplevel(0)
    tfidf_data = tfidf_data.sort_index().fillna(0)
    years = tfidf_data.index
    #print(years)
    cos_sim = 1-cosine_similarity(tfidf_data)
    cos_sim = min_max_scaler.fit_transform(cos_sim)
    sim_df = pd.DataFrame(cos_sim)
    sim_df.columns = years
    sim_df.index = years

    save_dict = {}

    for n, year in enumerate(years):
        if n < 2 or n > len(years)-2:
            pass
        else:
            before = years[:n]
            before_sum = calc_sum(sim_df, before)

            after = years[n:]
            after_sum = calc_sum(sim_df, after)

            weighted_avg = ((len(before)*before_sum) + (len(after)*after_sum))/(len(years))

            save_dict[year] = weighted_avg

    return save_dict, sim_df


if __name__ == '__main__':
    go()
