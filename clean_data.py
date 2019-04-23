from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

## Clean and Processing Data ##

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
    for k, v in speech_dict.items():
        if k not in clean_dict:
            clean_dict[k]= []
        for line in v:
            clean_dict[k].append([word.lower() for word in tokenizer.tokenize(line) if word not in stopWords])
    clean_dict = {k: [val for sublist in v for val in sublist] for k,v in clean_dict.items()}
    return clean_dict
