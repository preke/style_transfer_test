# coding = utf-8
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
import pickle
import math

# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


POS_LEXICON       = '../data/positive.txt'
NEG_LEXICON       = '../data/negative.txt'

class Preprocess(object):
    def __init__(self):
        self.english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'']  
        self.stop = set(stopwords.words('english'))

    def punctuate(self, text):
        ans = ""
        for letter in text:
            if letter in self.english_punctuations:
                ans += ' '
            else:
                ans += letter
        return ans

    def stop_removal(self, text):
        text = self.punctuate(text)
        word_list = word_tokenize(text)
        word_list = [i for i in word_list if i not in self.stop]
        return word_list

def get_wmd(list_pos, list_neg, pos_sentence_list, neg_sentence_list):
    logger.info('Train word2vec model...\n')
    list_all = list_pos + list_neg
    w2v_model = Word2Vec(list_all, min_count=5)

    sim_matrix = []
    for i in range(10):
        sim_matrix.append([])
        for j in range(len(list_neg)):
            sim_matrix[i].append(w2v_model.wmdistance(list_pos[i], list_neg[j]))

    logger.info('sim_matrix_shape: %d, %d\n' %(len(sim_matrix), len(sim_matrix[0])))

    with open('../data/wmd_result', 'w') as writer:
        for i in range(10):
            writer.write('Pos: %s\n' %pos_sentence_list[i])    
            #print sim_matrix[i]
            neg_index = int(np.argmin(np.array(sim_matrix[i])))
            #print neg_index
            writer.write('Neg: %s\n' %neg_sentence_list[neg_index]) 
            writer.write('\n')    


def read_in_data():
    list_pos = []
    list_neg = []
    pos_sentence_list = []
    neg_sentence_list = []
    preprocess = Preprocess()
    logger.info('Read in data...\n')
    with open('../data/pos.txt', 'r') as reader:
        for line in reader:
            pos_sentence_list.append(line)
            new_line = line.lower()
            new_line = preprocess.stop_removal(new_line)
            list_pos.append(new_line)

    with open('../data/neg.txt', 'r') as reader:
        for line in reader:
            neg_sentence_list.append(line)
            new_line = line.lower()
            new_line = preprocess.stop_removal(new_line)
            list_neg.append(new_line)

    return list_pos, list_neg, pos_sentence_list, neg_sentence_list



def mask_sentiment(list_pos, list_neg):
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

    list_pos_new = [['<neutral>' if lancaster_stemmer.stem(i) in pos_lex_set else i for i in word_list] for word_list in list_pos]
    list_neg_new = [['<neutral>' if lancaster_stemmer.stem(i) in neg_lex_set else i for i in word_list] for word_list in list_neg]
        
    return list_pos_new, list_neg_new

def mask_style_words(list_pos, list_neg):
    preprocess = Preprocess()
    # build style lexicons
    dict_pos = {}
    dict_neg = {}
    for line in list_pos:
        for word in line:
            try:
                dict_pos[word] += 1
            except:
                dict_pos[word] = 1
    for line in list_neg:
        for word in line:
            try:
                dict_neg[word] += 1
            except:
                dict_neg[word] = 1
       
    pos_words = []
    for k,v in dict_pos.items():
        try:
            tf_idf = float(v)*math.log(len(dict_neg)/(dict_neg[k] + 1))
        except:
            tf_idf = float(v)*math.log(len(dict_neg)/(0 + 1))
        pos_words.append((k, tf_idf))
    neg_words = []
    for k,v in dict_neg.items():
        try:
            tf_idf = float(v)*math.log(len(dict_pos)/(dict_pos[k] + 1))
        except:
            tf_idf = float(v)*math.log(len(dict_pos)/(0 + 1))
        neg_words.append((k, tf_idf))
    
    pos_words = sorted(pos_words, lambda x, y: cmp(x[1], y[1]), reverse=True)
    neg_words = sorted(neg_words, lambda x, y: cmp(x[1], y[1]), reverse=True)

    pos_words_set = set([i[0] for i in pos_words[:100]])
    neg_words_set = set([i[0] for i in neg_words[:100]])
    # print pos_words_set
    # print neg_words_set
    # mask in each sentence
    list_pos = [[j if j not in pos_words_set else '<neutral>' for j in i ] for i in list_pos ]
    list_neg = [[j if j not in neg_words_set else '<neutral>' for j in i ] for i in list_neg ]
    # return masked sentence list
    return pos_words_set, neg_words_set, list_pos, list_neg


if __name__ == '__main__':    
    list_pos, list_neg, pos_sentence_list, neg_sentence_list = read_in_data()
    list_pos_new, list_neg_new = mask_sentiment(list_pos, list_neg)
    get_wmd(list_pos_new, list_neg_new, pos_sentence_list, neg_sentence_list)




