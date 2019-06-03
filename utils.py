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
from operator import itemgetter
from os import listdir
from os.path import isfile, join
import pandas as pd
import textdistance
from itertools import groupby, chain
from collections import OrderedDict
stopWords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import utils_preprocessing as up
# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import utils as ut
import importlib
importlib.reload(ut)
from itertools import groupby
import yaml
import os
import textdistance

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
        df.to_csv("co_occurrence.csv", sep = ",")   
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
    Returns: list"""
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(strings)
    l = []
    for token in doc.noun_chunks:
        l.append(token)
    print(l)
    return l

def create_lda_objects(dictionary, year_break=1935, noun_phrase_analysis=True, paragraph_analysis=True, topics = 10):
    
    '''Creates gensim LDA objects
    Inputs:
        dictionary: dictionary with speeches. It comes from running "speeches, num_par = up.reading_data(PATH,'*.txt')"
        year_break(string): if "all", it creates a gensim LDA object for all speeches in the corpus. If not, it breaks
                            the corpus into two: speeches before and after that year (e.g. year_break = "1914")
        noun_phrase_analysis(boolean): if True, it creates a corpus using noun phrases. Otherwise, it uses words, and then 
        creates bags of words (or bag of noun phrases)
        paragraph_analysis (Boolean): 
        topics: number of topics to be modeled. Default is 10.
    Returns:
        gensim lda object if year_break=="all", if there is a break in the years, it returns a list of two gensim LDA objects

        '''
    
    if year_break!= 'all':
        dict1,dict2 = break_dictionary(dictionary, year_break)
        ls_of_dicts = [dict1,dict2]
    clean_dicts = []
    
    # NOUN PHRASES
    if noun_phrase_analysis:
        if year_break=='all':
            print ("Chunk function working")
            new_speeches = up.chunks(dictionary, 'regex') # Convert dictionary to noun phrases and get the paragraph number
            print ("Clean words working")
            clean_speeches = up.clean_words(new_speeches)
            print("Word changes working")
            words_changed = up.word_changes(clean_speeches, 0.5, 100)
            print("phrases_lemmed working")
            phrases_lemmed = up.lemmed_phrases(words_changed, clean_speeches)
            if paragraph_analysis:
                documents = ut.convert_dict_to_list(phrases_lemmed)
            else:
                documents = convert_dict_to_lemmedNF_to_LS_SPEECH (phrases_lemmed)
        else:
            for dict_break in ls_of_dicts:
                print ("Chunk function working")
                new_speeches = up.chunks(dict_break, 'regex') # Convert dictionary to noun phrases and get the paragraph number
                print ("Clean words working")
                clean_speeches = up.clean_words(new_speeches)
                print("Word changes working")
                words_changed = up.word_changes(clean_speeches, 0.5, 100)
                print("phrases_lemmed working")
                phrases_lemmed = up.lemmed_phrases(words_changed, clean_speeches)
                clean_dicts.append(phrases_lemmed)
            if paragraph_analysis:
                documents=[]
                for dict_break2 in clean_dicts:
                    document = ut.convert_dict_to_list(dict_break2)
                    documents.append(document)
            else:
                documents=[]
                for dict_break2 in clean_dicts:
                    document = convert_dict_to_lemmedNF_to_LS_SPEECH(dict_break2)
                    documents.append(document)
    # WORDS                
    else:
        documents=[]
        if year_break == "all":
            print("cleaning data")
            data = clean_words_by_paragraph_LDA(dictionary)
            if paragraph_analysis:
                for k,v in data.items():
                    for paragraph in v:
                        documents.append(v)
            else:
                for k,v in data.items():
                    one_speech=[]
                    for paragraph in v:
                        one_speech.extend(paragraph)
                    documents.append(one_speech)   
        else:
            if paragraph_analysis:
                for dict_break2 in ls_of_dicts:
                    ls=[]
                    data = clean_words_by_paragraph_LDA(dict_break2)
                    for k,v in data.items():
                        for paragraph in v:
                            ls.append(v)
                    documents.append(ls)
            else:
                 for dict_break2 in ls_of_dicts:
                    data = clean_words_by_paragraph_LDA(dict_break2)
                    ls=[]
                    for k,v in data.items():
                        one_speech = []
                        for paragraph in v:
                            one_speech.extend(paragraph)
                        ls.append(one_speech)
                    documents.append(ls)
    if year_break=="all":
        id2word = corpora.Dictionary(documents)
        corpus = [ id2word.doc2bow(text) for text in documents]
        corpora.MmCorpus.serialize('corpus.mm', corpus)
        once_ids = [tokenid for tokenid, docfreq in id2word.dfs.items() if docfreq== 1]

        print(id2word)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                        num_topics=topics,
                                           random_state=100,
                                           passes=2,
                                           alpha='auto',
                                           per_word_topics=True)
    else:
        lda_model=[]
        i=0
        for document in documents:
            id2word = corpora.Dictionary(document)
            corpus = [ id2word.doc2bow(text) for text in document]
            corpora.MmCorpus.serialize('corpus'+str(i)+'.mm', corpus)
            #once_ids = [tokenid for tokenid, docfreq in id2word.dfs.items() if docfreq== 1]
            lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                        num_topics=topics,
                                           random_state=100,
                                           passes=2,
                                           alpha='auto',
                                           per_word_topics=True)
            lda_model.append(lda)
            i=i+1
            
    return lda_model

def create_lda_viz(lda_object, corpusmm):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_object, corpora.MmCorpus(corpusmm), lda_object.id2word)
    return vis

def convert_dict_to_lemmedNF_to_LS_SPEECH(dictionary):
    '''Creates a list of speeches in which each list is a "document"
    with all noun phrases by speech'''
    
    list_document=[]

    for key, value in dictionary.items():
        ls = []
        for speech in value:
            ls.append(speech[0])
        list_document.append(ls)
    return list_document

def break_dictionary(dictionary,yearbreak):
    
    break1= {}
    break2 = {}
    for k,v in dictionary.items():
        if int(k[1])<=int(yearbreak):
            break1[k] = v
        else:
            break2[k] = v
    return break1,break2
def num_there(s):
    return any(i.isdigit() for i in s)

def clean_words_by_paragraph_LDA(speech_dict):
    """
    Creates a dictionary which keys are the
    tuple (president,year), and its value a list of
    tokenized words without StopWords.
    
    Inputs: Dictionary with speeches
    Returns: Dictionary
    """

    clean_dict = {}
    tokenizer = RegexpTokenizer(r'\w+')
    for k, v in speech_dict.items():
        if k not in clean_dict:
            clean_dict[k]= []
        ls=[]
        for paragraph in v:
            #print(paragraph)
            one_doc=[]
            new_paragr = tokenizer.tokenize(paragraph)
            #print(line)
            #print(line)
            for word in new_paragr:
                word = word.lower()
                if num_there(word)== False and word not in stopWords and len(word) >0:
                    one_doc.append(word)
            ls.append(one_doc)
            clean_dict[k].append(ls) 
    clean_dict = {k: [val for sublist in v for val in sublist] for k,v in clean_dict.items()}
    return clean_dict

if __name__ == '__main__':
    go()
