import re
import os
import torch
import torch.nn as nn
import torchtext.data as data
import torchtext.datasets as datasets
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
import pickle
import numpy as np
import codecs
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import defaultdict, Counter, OrderedDict

POS_LEXICON       = '../data/positive.txt'
NEG_LEXICON       = '../data/negative.txt'

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def expierment_name(args, ts):

    exp_name = str()
    exp_name += "BS=%i_"%args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_"%args.embedding_size
    exp_name += "%s_"%args.rnn_type.upper()
    exp_name += "HS=%i_"%args.hidden_size
    exp_name += "L=%i_"%args.num_layers
    exp_name += "BI=%i_"%args.bidirectional
    exp_name += "LS=%i_"%args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_"%args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_"%args.x0
    exp_name += "TS=%s"%ts

    return exp_name

def preprocess(in_path, pos_output_paths, neg_output_paths):
    '''
    Remove labels and split data into 2 files(.pos and .neg)
    '''
    pos_writer = open(pos_output_paths, 'w')
    neg_writer = open(neg_output_paths, 'w')
    with open(in_path, 'r') as reader:
        for line in reader:
            if line[9] == '1': # negative
                text = line[11:]
                # neg_writer.write(text.split(': ')[0].lower()) # title
                # neg_writer.write('\t')
                neg_writer.write(text.split(': ')[1].lower())
            if line[9] == '2': # positive
                text = line[11:]
                # pos_writer.write(text.split(': ')[0].lower()) # title
                # pos_writer.write('\t') 
                pos_writer.write(text.split(': ')[1].lower())
    pos_writer.close()
    neg_writer.close() 

def preprocess_pos_neg(in_path, out_path):
    writer = open(out_path, 'w')
    with open(in_path, 'r') as reader:
        for line in reader:
            writer.write(line.split('\t')[1].lower())
    
    writer.close()


def preprocess_write(in_path, out_path):
    '''
    Generate tsv file to load easier
    '''
    writer = open(out_path, 'w')
    with open(in_path, 'r') as reader:
        for line in reader:
            if line[9] == '1': # negative
                text = line[11:]
                writer.write('0\t')
                writer.write(text.split(': ')[1].lower())
            if line[9] == '2': # positive
                text = line[11:]
                writer.write('1\t')
                writer.write(text.split(': ')[1].lower())                
    writer.close()

def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec

def get_pretrained_word_embed(glove_path, args, text_field):
    embedding_dict = load_glove_as_dict(glove_path)
    word_vec_list = []
    for idx, word in enumerate(text_field.vocab.itos):
        if word in embedding_dict:
            try:
                vector = np.array(embedding_dict[word], dtype=float).reshape(1, args.embed_dim)
            except:
                vector = np.random.rand(1, args.embed_dim)
        else:
            vector = np.random.rand(1, args.embed_dim)
        word_vec_list.append(torch.from_numpy(vector))
    wordvec_matrix = torch.cat(word_vec_list)
    return wordvec_matrix


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_pos_neg_rep(word_2_index, pretrained_weight):
    pos_lex_list = []
    neg_lex_list = []
    with open(POS_LEXICON, 'r') as reader:
        for line in reader:
            pos_lex_list.append(word_tokenize(line)[0])

    with open(NEG_LEXICON, 'r') as reader:
        for line in reader:
            neg_lex_list.append(word_tokenize(line)[0])

    pos_lex_list = [pretrained_weight[word_2_index[i]] for i in pos_lex_list]
    neg_lex_list = [pretrained_weight[word_2_index[i]] for i in neg_lex_list]
    print(torch.stack(pos_lex_list, dim=0).size())
    print(torch.stack(pos_lex_list, dim=1).size())
    # for i in pos_lex_list:
    #     print(i.size())
    return 0,0






