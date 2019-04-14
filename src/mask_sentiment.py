'''
    Mask sentiment words use sentiment lexicons
    Generate *.pos_mask
'''
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging
import logging.config


POS_LEXICON       = '../data/positive.txt'
NEG_LEXICON       = '../data/negative.txt'

TRAIN_PATH        = '../data/amazon_train.tsv'
TEST_PATH         = '../data/amazon_test.tsv'
MASKED_TRAIN_PATH = '../data/mask_amazon_train.tsv'
MASKED_TEST_PATH  = '../data/mask_amazon_test.tsv'

def mask():
    lancaster_stemmer = LancasterStemmer()
    pos_lex_list = []
    neg_lex_list = []
    with open(POS_LEXICON, 'r') as reader:
        for line in reader:
            pos_lex_list.append(lancaster_stemmer.stem(word_tokenize(line)[0]))

    with open(NEG_LEXICON, 'r') as reader:
        for line in reader:
            neg_lex_list.append(lancaster_stemmer.stem(word_tokenize(line)[0]))

    pos_lex_set = set(pos_lex_list)
    neg_lex_set = set(neg_lex_list)

    # test_writer = open(MASKED_TEST_PATH, 'w')
    print(pos_lex_set)
    # with open(TEST_PATH, 'r') as reader:
    #     for line in reader:
    #         list_ = line.split('\t')
    #         if list_[0] == '1': # positive
    #             word_list = word_tokenize(list_[1])
    #             word_list = [lancaster_stemmer.stem(i) for i in word_list]
    #             word_list = ['<PAD>' if i in pos_lex_set else i for i in word_list]
    #         if list_[0] == '0': # negative
    #             word_list = word_tokenize(list_[1])
    #             word_list = [lancaster_stemmer.stem(i) for i in word_list]
    #             word_list = ['<PAD>' if i in neg_lex_set else i for i in word_list]

    #         print(word_list)

mask()


