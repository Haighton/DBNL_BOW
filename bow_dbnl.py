#!/usr/bin/env python3
"""Generate Bag-of-Words from dataset of TEI-XML files from DBNL.

os.walk through sys.argv[1](path of DBNL Dataset) and look for all files that
end with '.xml'. Pick a subset from all XML files, extract all relevant text.
Tokenize text into sentences and clean up. Tokenize sentences into words and
generate bag-of-words. Write bow to file.

Originally written to create a custom language dictionary for spell checking
historical dutch newspaper article titles. For that purpose it is fine to use
random files from the dataset as input.
Comment-out the call to `pick_random` to use a whole dateset.
(DOCS_USED is ignored)

Download `DBNL Dataset`_
    .. _DBNL Dataset: https://dbnl.org/letterkunde/pd/index.php

Usage::
    $ python3 bow_dbnl.py 'path/to/Dataset'

    Before starting the script set the following globar vars below
    - Set BOW_LENGTH global variable (0 = max length).
    - Set DOCS_USED: Number of files used from dataset.
    - Set OUTPUT_TXT: location of output bow txt.

Output:
    TSV text file with on each row: word \t frequency
    The amount of rows depend on the BOW_LENGTH variable.
    The resulting .txt file is saved at OUTPUT_TXT.

TODO:
    * Show which files have been chosen? in log?
    * Language check? (I think I saw french sentences)

"""
import sys
import os
import re
from collections import OrderedDict
import random

import nltk

# Total amount of words in Bag-of-words.
# 0 is max length (each unique word found in dataset - you probably don't want
# this, as this can easily surpass a milions words).
# If user BOW_LENGTH > unique words, BOW_LENGTH is set to number of unique
# words.
# The optimal value depends on the value of DOCS_USED (using more documents
# will result in a greater chance of finding more unique words), user
# preferrence or specific required length because of restrictions. The longer
# the BoW, the longer it takes to write every word and frequency to ouput file
# (I haven't tried values higher than 50000).
BOW_LENGTH = 50000

# TEI files are random.shuffled and slice is taken from start to DOCS_USED
# (>1000 and process takes really long. Example output range: 2.352.233 unique
# words in 3.000 documents)
DOCS_USED = 3000

# BOW txt output location (r'./dbnl_bow-100_10docs.txt').
OUTPUT_TXT = r'../output/dbnl_bow-50k_3kdocs.txt'

# Set Random seed for input file shuffle. Used in pick_random().
random.seed(4)


def get_files(path_batch):
    """Get paths of TEI files.

    Travers `path_batch` and look for filenames ending in '.xml' and save their
    path into a list called `tei_files`.

    Arguments:
        path_batch (str): Path to batch.

    Returns:
        list: List with paths to each individal TEI file found in [path_batch].
    """
    tei_files = []
    print('\nLooking for files:')
    for path_dir, dirnames, filenames in os.walk(path_batch):
        for filename in filenames:
            if filename.endswith('.xml'):
                print(f'{filename}', end='\r', flush=True)
                tei_files.append(os.path.join(path_dir, filename))
    print(f'\n\nFound {len(tei_files)} XML files.')
    return(tei_files)


def extract_tei(tei_files):
    """Extract flat text from TEI-XML.

    Split TEI document at <body> tag and only take text from the body.
    Text is then tokenized into sentences and cleaned up:
        - XML tags removed (all relevant text is text() value of all elements).
        - non-alphanumeric characters removed.
        - extra whitespace removed.

    Arguments:
        tei_files (list): List with paths to TEI-XML files.

    Returns:
        list: List of sentences.
    """
    flat_text = []
    count = 0
    print('\n\nExtracting TEI-XML:')
    for tei_file in tei_files:
        count += 1
        print(f'[{round(count/len(tei_files)*100.0, 1)}%] Extracting sentences from {os.path.basename(tei_file)}',
              end='\r', flush=True)
        with open(tei_file, encoding='utf8') as tei_doc:
            tei = tei_doc.read()

            # Split at <body>, we only need the body text of the file.
            tei_body = re.split(r'<body>', tei)

            # Remove all XML tags.
            tei_ptxt_noxml = re.sub(r'<[^<]+>', '', tei_body[1]
                                    .replace('\n', ' '))

            # Tokenize page text into sentences and clean text.
            txt_clean = nltk.sent_tokenize(tei_ptxt_noxml.lower())
            for i in range(len(txt_clean)):
                txt_clean[i] = re.sub('&nbsp;', ' ', txt_clean[i])
                txt_clean[i] = re.sub('&amp;', ' ', txt_clean[i])
                txt_clean[i] = re.sub(r'\W', ' ', txt_clean[i])
                txt_clean[i] = re.sub(r'\s+', ' ', txt_clean[i])
                flat_text.append(txt_clean[i])
    return(flat_text)


def create_bow(flat_text):
    """Create Bag-of-Words.

    Arguments:
        flat_text (list): Sentences found in all the TEI files.

    Returns:
        dict: bag-of-words as OrderedDict
    """
    dict_wordfreq = OrderedDict()
    count = 0
    print('\n\nCreating Bag-Of-Words:')
    for sen in flat_text:
        words = nltk.word_tokenize(sen)  # Split sentences into words.
        count += 1
        for word in words:
            if len(word) > 2:  # Word has more than 2 characters.
                if word not in dict_wordfreq.keys():
                    dict_wordfreq[word] = 1
                else:
                    dict_wordfreq[word] += 1
                print(f'[{round(count/len(flat_text)*100.0, 1)}%]',
                      end='\r', flush=True)

    # Sort words by frequency (highest first).
    dict_sorted = OrderedDict(sorted(dict_wordfreq.items(), key=lambda x: x[1],
                              reverse=True))
    return(dict_sorted)


def output_bow(dict_sorted, BOW_LENGTH, OUTPUT_TXT):
    """Write bow with lenth BOW_LENGTH to OUTPUT_TXT."""
    if BOW_LENGTH <= 0 or BOW_LENGTH > len(dict_sorted):
        BOW_LENGTH = len(dict_sorted)

    print(f'\n\nTotal amount of unique words found: {len(dict_sorted)}')
    print(f'\nWriting top {BOW_LENGTH} most frequent words to file:\
          \n{OUTPUT_TXT}')

    # Write to txt file.
    with open(OUTPUT_TXT, 'w') as outputf:
        for i in range(0, BOW_LENGTH):
            outputf.write(f'{list(dict_sorted.keys())[i]}\t{list(dict_sorted.values())[i]}\n')


def pick_random(tei_files, DOCS_USED):
    """Random.shuffle tei_files and take slice from start to DOCS_USED."""
    if DOCS_USED <= 0 or DOCS_USED > len(tei_files):
        DOCS_USED = len(tei_files)
    random.shuffle(tei_files)
    print(f'Taking {DOCS_USED} random files.')
    return(tei_files[:DOCS_USED])


if __name__ == '__main__':
    tei_files = get_files(sys.argv[1])

    # Turn off to use full dataset.
    tei_files = pick_random(tei_files, DOCS_USED)

    flat_text = extract_tei(tei_files)
    dict_sorted = create_bow(flat_text)
    output_bow(dict_sorted, BOW_LENGTH, OUTPUT_TXT)

    print('\n\nDONE!\n')
