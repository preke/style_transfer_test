# coding = utf-8
import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
import os
import datetime
import time
import traceback
import model
from gensim.models import Word2Vec


# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# self define
from utils import preprocess_write, get_pretrained_word_embed, preprocess_pos_neg, get_pos_neg_rep
from dataload import load_data, load_pos_neg_data
from model import SentenceVAE, CNN_Text
from train import train_vae, train_cnn, eval_vae, test_vae


# paths

GLOVE_PATH        = '../data/glove.6B.300d.txt'

mask_amazon_train = '../data/amazon/train.mask'
mask_amazon_test  = '../data/amazon/test.mask'

mask_yelp_train   = '../data/yelp/new_train.mask'
mask_yelp_test    = '../data/yelp/new_test.mask'
mask_yelp_ref     = '../data/yelp/reference.mask'


parser = argparse.ArgumentParser(description='')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-train', action='store_true', default=True, help='train or test')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-cnn_snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()


# Parameters setting
args.grad_clip    = 2
args.embed_dim    = 300
args.hidden_dim   = 200
args.batch_size   = 32
args.lr           = 0.0005
args.num_epoch    = 1000
args.max_length   = 20
args.device       = torch.device('cuda')
args.dropout      = 0.1


args.rnn_type          = 'gru'
args.word_dropout      = 0.0
args.embedding_dropout = 0.0
args.latent_size       = 16
args.num_layers        = 1
args.bidirectional     = True
args.anneal_function   = 'logistic'
args.k                 = 0.0025
args.x0                = 2500
args.print_every       = 100


## CNN classifier

args.kernel_num    = 300
args.kernel_sizes  = '3'
args.kernel_sizes  = [int(k) for k in args.kernel_sizes.split(',')]
args.cnn_save_dir  = './cnn/'
args.num_class     = 2
args.cnn_lr        = 0.0001
args.early_stop    = 3000

args.test_interval = 100
args.save_interval = 1000
args.log_interval  = 1
args.cuda          = True
args.epochs        = 10
args.save_best     = True

# Load data
logger.info('Loading data begin...')
text_field, label_field, train_data, train_iter, eval_data, eval_iter, test_data, test_iter = load_data(mask_yelp_train, mask_yelp_test, mask_yelp_ref, args)
text_field.build_vocab(train_data, eval_data, test_data, min_freq=5)
label_field.build_vocab(train_data)
logger.info('Length of vocab is: ' + str(len(text_field.vocab)))



args.vocab_size   = len(text_field.vocab)
args.word_2_index = text_field.vocab.stoi # tuple of dict({word: index})
args.index_2_word = text_field.vocab.itos # only list of words
args.text_field   = text_field

# print(args.word_2_index['<SOS>']) # 2
# print(args.word_2_index['<EOS>']) # 3
# print(args.word_2_index['<PAD>']) # 1
# print(args.word_2_index['<UNK>']) # 0

# Initial word embedding
logger.info('Getting pre-trained word embedding ...')
args.pretrained_weight = get_pretrained_word_embed(GLOVE_PATH, args, text_field)  
print(type(args.pretrained_weight))
args.pos_rep, args.neg_rep = get_pos_neg_rep(args.word_2_index, args.pretrained_weight)

## Build CNN sentiment classifier
args.cnn_snapshot = './cnn/yelp.pt'
# args.cnn_snapshot = './cnn/test_best_steps_900.pt'
cnn = model.CNN_Text(args)
if args.cnn_snapshot is not None:
    logger.info('Load CNN classifier from' + args.cnn_snapshot)
    cnn.load_state_dict(torch.load(args.cnn_snapshot))
    cnn = cnn.cuda()
else:
    logger.info('Train CNN classifier begin...')
    try:
        train_cnn(train_iter=train_iter, eval_iter=test_iter, model=cnn, args=args)
    except KeyboardInterrupt:
        print(traceback.print_exc())
        print('\n' + '-' * 89)
        print('Exiting from training early')


# Build Sentence_VAE model and train



logger.info('Build VAE model...')
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
    pre_embedding       = args.pretrained_weight,
    args                = args)

vae_model = vae_model.cuda()
if args.snapshot is not None:
    logger.info('Load model from' + args.snapshot)
    vae_model.load_state_dict(torch.load(args.snapshot))
    vae_model = vae_model.cuda()


    train_vae(train_iter=test_iter, eval_iter=test_iter, model=vae_model, args=args, sentiment_classifier=cnn)
    
    # w2v_model = Word2Vec.load("yelp_word2vec.model")
    # test_vae(model                = vae_model,
    #          test_iter            = test_iter,
    #          args                 = args,
    #          sentiment_classifier = cnn,
    #          w2v_model            = w2v_model)

else:
    logger.info('Train model begin...')
    try:
        # cnn.eval()
        train_vae(train_iter=train_iter, eval_iter=eval_iter, model=vae_model, args=args, sentiment_classifier=cnn)
    except KeyboardInterrupt:
        print(traceback.print_exc())
        print('\n' + '-' * 89)
        print('Exiting from training early')








