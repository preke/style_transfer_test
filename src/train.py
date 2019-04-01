import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets

import codecs
from nltk.corpus import sentiwordnet as swn
import pickle
import numpy as np
import pandas as pd
import codecs
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import traceback

import dataload
from model import *
from utils import preprocess_write, get_pretrained_word_embed, preprocess_pos_neg
from dataload import load_data, load_pos_neg_data



# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

best_results = 0

def train_S2S(train_iter, dev_iter, train_data, model, args):
    save_dir = "RGLModel/Newdata/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    optimizer        = optim.Adam(model.parameters(), lr=args.lr)
    loss_reconstruct = nn.NLLLoss()
    n_epoch          = args.num_epoch
    lamda            = args.lamda
    len_iter         = int(len(train_data)/args.batch_size) + 1
    cnt_epoch = 0
    cnt_batch = 0
    for epoch in range(n_epoch): 
        logger.info('In ' + str(cnt_epoch) + ' epoch... ')
        for batch in train_iter:
            model.train()
            sample  = batch.text[0]
            length  = batch.text[1]
            p       = float(cnt_batch + epoch * len_iter) / n_epoch / len_iter
            alpha   = 2. / (1. + np.exp(-10 * p)) - 1
            feature = Variable(sample)

            reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()], feature[:, :-1])
            feature_iow     = Variable(feature[:,1:].contiguous().view(-1)).cuda()

            optimizer.zero_grad()
            reconstruct_loss = loss_reconstruct(reconstruct_out, feature_iow)
            err = reconstruct_loss
            err.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()
            if cnt_batch % 100 == 0:
                # logger.info('Train_loss:{:.6f}'.format(reconstruct_loss))
                avg_loss = eval_S2S(dev_iter, model)
            if cnt_batch % 1000 == 0:                
                show_reconstruct_results_S2S(dev_iter, model, args, cnt_batch, avg_loss)
            cnt_batch += 1
        cnt_epoch += 1

def eval_S2S(dev_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for batch in dev_iter:
        sample  = batch.text[0]
        length  = batch.text[1]
        feature = Variable(sample)
        
        reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()])
        feature_iow      = Variable(feature.contiguous().view(-1)).cuda() # the whole sentence

        loss_reconstruct = nn.NLLLoss()
        reconstruct_loss = loss_reconstruct(reconstruct_out, feature_iow)


        avg_loss += reconstruct_loss.data
        size += len(sample)

    avg_loss /= size
    logger.info('Evaluation - Train_loss:{:.6f}, eva_loss: {:.6f}\n'.format(reconstruct_loss, avg_loss))
    return avg_loss

def show_reconstruct_results_S2S(dev_iter, model, args, cnt, avg_loss):
    writer = open('s2s_logs_'+str(cnt) + '__' + str(float(avg_loss)) + '_.txt', 'w')
    cnt_batch = 0
    for batch in dev_iter:
        sample  = batch.text[0]
        length  = batch.text[1]
        feature = Variable(sample)
        
        reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()])
        out_in_batch = reconstruct_out.contiguous().view(len(length), args.max_length, args.vocab_size)
        k = 0 
        for i in out_in_batch:
            writer.write(' '.join([args.index_2_word[int(l)] for l in sample[k]]))
            writer.write('\n=============\n')
            writer.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(i, dim=-1)]))
            writer.write('\n************\n')
            k = k + 1
        cnt_batch += 1
    writer.close()

