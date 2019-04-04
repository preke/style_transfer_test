# coding = utf-8
import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
import os
import datetime
import traceback
import model



# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# self define
from utils import preprocess_write, get_pretrained_word_embed, preprocess_pos_neg
from dataload import load_data, load_pos_neg_data
from model import Seq2Seq, SentenceVAE
from train import eval_S2S, train_S2S, show_reconstruct_results_S2S, train_vae


# paths
TRAIN_PATH     = '../data/train.ft.txt'
TEST_PATH      = '../data/test.ft.txt'
TEST_PRE_PATH  = '../data/t.tsv'
POS_TEST_PATH  = '../data/test.pos'
NEG_TEST_PATH  = '../data/test.neg'
POS_TRAIN_PATH = '../data/train.pos'
NEG_TRAIN_PATH = '../data/train.neg'
GLOVE_PATH     = '../data/glove.6B.300d.txt'

small_pos_path   = '../data/amazon_small.pos'
small_neg_path   = '../data/amazon_small.neg'
small_pos        = '../data/amazon_small.pos'
small_neg        = '../data/amazon_small.neg'

small_glove_path = '../data/wordvec.txt'
small_path       = '../data/small.txt'
small_pre_path   = '../data/small_preprocess.tsv'


amazon_train = '../data/amazon_train.tsv'
amazon_test = '../data/amazon_test.tsv'

parser = argparse.ArgumentParser(description='')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-train', action='store_true', default=True, help='train or test')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()


# Parameters setting
args.grad_clip    = 2
args.embed_dim    = 300
args.hidden_dim   = 100
args.batch_size   = 32
args.lr           = 0.0001
args.num_epoch    = 200
args.num_class    = 2
args.max_length   = 20
args.device       = torch.device('cuda')
args.kernel_num   = 100
args.kernel_sizes = '3,4,5'
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.dropout      = 0.1


args.rnn_type          = 'gru'
args.word_dropout      = 0
args.embedding_dropout = 0.5
args.latent_size       = 16
args.num_layers        = 1
args.bidirectional     = True
args.anneal_function   = 'logistic'
args.k                 = 0.0025
args.x0                = 2500
args.print_every       = 50


# Preprocess
if not os.path.exists(TEST_PRE_PATH):
    logger.info('Preprocessing begin...')
    preprocess_write(TEST_PATH, TEST_PRE_PATH)
else:
    logger.info('No need to preprocess!')


# Load data
logger.info('Loading data begin...')
text_field, label_field, train_data, train_iter, dev_data, dev_iter = load_data(amazon_test, amazon_test, args)
text_field.build_vocab(train_data, dev_data, min_freq=5)
label_field.build_vocab(train_data)
logger.info('Length of vocab is: ' + str(len(text_field.vocab)))


args.vocab_size = len(text_field.vocab)
args.word_2_index = text_field.vocab.stoi # tuple of dict({word: index})
args.index_2_word = text_field.vocab.itos # only list of words

# Initial word embedding
logger.info('Getting pre-trained word embedding ...')
args.pretrained_weight = get_pretrained_word_embed(GLOVE_PATH, args, text_field)  



# python main.py -snapshot RGLModel/epoch_10_batch_254000_acc_85.2_bestmodel.pt


'''
# Build s2s model and train
s2s_model = Seq2Seq(src_nword=args.vocab_size, 
                    trg_nword=args.vocab_size, 
                    num_layer=2, 
                    embed_dim=args.embed_dim, 
                    hidden_dim=args.hidden_dim, 
                    max_len=args.max_length, 
                    trg_soi=args.word_2_index['<SOS>'], 
                    args=args)
s2s_model.cuda()
if args.snapshot is not None:
    logger.info('Load model from' + args.snapshot)
    s2s_model.load_state_dict(torch.load(args.snapshot))
    show_reconstruct_results(dev_iter, s2s_model, args)
    # if not os.path.exists(small_pos):
    #     preprocess_pos_neg(small_pos_path, small_pos)
    #     preprocess_pos_neg(small_neg_path, small_neg)
    # pos_iter, neg_iter = load_pos_neg_data(small_pos, small_neg, text_field, args)
    # style_transfer(pos_iter, neg_iter, rgl_net, args)
else:
    logger.info('Train model begin...')

    try:
        train_S2S(train_iter=train_iter, dev_iter=dev_iter, train_data=train_data, model=s2s_model, args=args)
    except KeyboardInterrupt:
        print(traceback.print_exc())
        print('\n' + '-' * 89)
        print('Exiting from training early')
'''

# Build Sentence_VAE model and train

vae_model = SentenceVAE(
    vocab_size          = args.vocab_size,
    sos_idx             = args.word_2_index['<SOS>'],
    eos_idx             = args.word_2_index['<EOS>'],
    pad_idx             = args.word_2_index['<PAD>'],
    unk_idx             = args.word_2_index['<UNK>'],
    max_sequence_length = args.max_length,
    embedding_size      = args.embed_dim,
    rnn_type            = args.rnn_type,
    hidden_size         = args.hidden_dim,
    word_dropout        = args.word_dropout,
    embedding_dropout   = args.embedding_dropout,
    latent_size         = args.latent_size,
    num_layers          = args.num_layers,
    bidirectional       = args.bidirectional,
    pre_embedding       = args.pretrained_weight)

vae_model = vae_model.cuda()
if args.snapshot is not None:
    logger.info('Load model from' + args.snapshot)
    vae_model.load_state_dict(torch.load(args.snapshot))
else:
    logger.info('Train model begin...')
    try:
        train_vae(train_iter=train_iter, eval_iter=dev_iter, model=vae_model, args=args)
    except KeyboardInterrupt:
        print(traceback.print_exc())
        print('\n' + '-' * 89)
        print('Exiting from training early')








