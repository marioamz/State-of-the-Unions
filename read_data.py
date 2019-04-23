# Script for reading in the data

import glob
import os
import re
import sys


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


if __name__ == '__main__':
    go()
