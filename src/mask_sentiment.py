'''
    Mask sentiment words use sentiment lexicons
    Generate *.pos_mask
'''
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import logging
import logging.config


POS_LEXICON       = '../data/positive.txt'
NEG_LEXICON       = '../data/negative.txt'

TRAIN_PATH        = '../data/amazon_train.tsv'
TEST_PATH         = '../data/amazon_test.tsv'
MASKED_TRAIN_PATH = '../data/mask_amazon_train.tsv'
MASKED_TEST_PATH  = '../data/mask_amazon_test.tsv'

def mask():
    pos_lex_list = []
    neg_lex_list = []
    with open(POS_LEXICON, 'r') as reader:
        for line in reader:
            pos_lex_list.append(line)

    with open(NEG_LEXICON, 'r') as reader:
        for line in reader:
            neg_lex_list.append(line)

    pos_lex_list = set(pos_lex_list)
    neg_lex_list = set(neg_lex_list)

    # test_writer = open(MASKED_TEST_PATH, 'w')
    with open(TEST_PATH, 'r') as reader:
        for line in reader:
            print line.split('\t')
            break
            # test_writer.



