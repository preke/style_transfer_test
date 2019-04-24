'''
Mask sentiment words in train and test data use sentiment lexicons
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

# TRAIN_PATH        = '../data/amazon/train.tsv'
# TEST_PATH         = '../data/amazon/test.tsv'
# MASKED_TRAIN_PATH = '../data/amazon/train.mask'
# MASKED_TEST_PATH  = '../data/amazon/test.mask'

# TRAIN_PATH        = '../data/yelp/train.tsv'
# TEST_PATH         = '../data/yelp/test.tsv'
# MASKED_TRAIN_PATH = '../data/yelp/train.mask'
# MASKED_TEST_PATH  = '../data/yelp/test.mask'



reference_PATH         = '../data/yelp/reference.tsv'
MASKED_reference_PATH  = '../data/yelp/reference.mask'

def punctuate(text):
    ans = ""
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '`']
    for letter in text:
        if letter in english_punctuations:
            ans += ' '
        else:
            ans += letter
    return ans

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

    test_writer = open(MASKED_TEST_PATH, 'w')
    with open(TEST_PATH, 'r') as reader:
        for line in reader:
            list_ = line.split('\t')
            word_list = word_tokenize(punctuate(list_[1].lower()))
            tmp_wlist = word_list
            if list_[0] == '1': # positive
                word_list = ['<pos>' if lancaster_stemmer.stem(i) in pos_lex_set else i for i in word_list]
            if list_[0] == '0': # negative
                word_list = ['<neg>' if lancaster_stemmer.stem(i) in neg_lex_set else i for i in word_list]
            test_writer.write(list_[0] + '\t' + ' '.join(tmp_wlist) + '\t' + ' '.join(word_list) + '\n')
    test_writer.close()

    train_writer = open(MASKED_TRAIN_PATH, 'w')
    with open(TRAIN_PATH, 'r') as reader:
        for line in reader:
            list_ = line.split('\t')
            word_list = word_tokenize(punctuate(list_[1]))
            tmp_wlist = word_list
            if list_[0] == '1': # positive
                word_list = ['<pos>' if lancaster_stemmer.stem(i) in pos_lex_set else i for i in word_list]
            if list_[0] == '0': # negative
                word_list = ['<neg>' if lancaster_stemmer.stem(i) in neg_lex_set else i for i in word_list]
            # train_writer.write(list_[0] + '\t' + ' '.join(word_list) + '\n')
            train_writer.write(list_[0] + '\t' + ' '.join(tmp_wlist) + '\t' + ' '.join(word_list) + '\n')
    train_writer.close()

def mask_ref():
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

    reference_writer = open(MASKED_reference_PATH, 'w')
    with open(reference_PATH, 'r') as reader:
        for line in reader:
            list_ = line.split('\t')
            word_list = word_tokenize(punctuate(list_[1].lower()))
            tmp_wlist = word_list
            if list_[0] == '0': # positive
                word_list = ['<pos>' if lancaster_stemmer.stem(i) in pos_lex_set else i for i in word_list]
            if list_[0] == '1': # negative
                word_list = ['<neg>' if lancaster_stemmer.stem(i) in neg_lex_set else i for i in word_list]
            reference_writer.write(list_[0] + '\t' + ' '.join(word_list) + '\t' + list_[2])
    reference_writer.close()



mask_ref()


