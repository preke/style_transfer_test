import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets
import time
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

from collections import OrderedDict, defaultdict



# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

best_results = 0

def train_S2S(train_iter, dev_iter, train_data, model, args):
    save_dir = "../model/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    optimizer        = optim.Adam(model.parameters(), lr=args.lr)
    loss_reconstruct = nn.NLLLoss()
    n_epoch          = args.num_epoch
    len_iter         = int(len(train_data)/args.batch_size) + 1
    cnt_epoch = 0
    cnt_batch = 0
    for epoch in range(n_epoch): 
        logger.info('In ' + str(cnt_epoch) + ' epoch... ')
        for batch in train_iter:
            model.train()
            sample  = batch.text[0]
            length  = batch.text[1]
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
    writer = open('res/s2s_logs_'+str(cnt) + '__' + str(float(avg_loss)) + '_.txt', 'w')
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



def eval_vae(model, eval_iter, args, step, cur_epoch, iteration):
    model.eval()
    writer = open('res/vae_epoch_'+str(cur_epoch) + '_batch_' + str(iteration) + '_.txt', 'w')
    for batch in eval_iter:
        sample     = batch.text[0]
        length     = batch.text[1]
        length     = torch.add(length, -1)
        batch_size = len(sample)
        feature    = Variable(sample)
        _input     = feature[:, :-1]
        target     = feature[:, 1:]
        
        _, mean, logv, z = model(_input, length)
        generations, z = model.inference(z)
        generations, z = model.inference(z)
        print(generations)
        # NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
        #     length, mean, logv, args.anneal_function, step, args.k, args.x0, model.pad_idx)

        # loss = (NLL_loss + KL_weight * KL_loss)/batch_size
        # # print("Valid: Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
        # #         %(loss.data[0], NLL_loss.data[0]/batch_size, KL_loss.data[0]/batch_size, KL_weight))

        # k = 0 
        # for i in logp:
        #     writer.write(' '.join([args.index_2_word[int(l)] for l in sample[k]]))
        #     writer.write('\n=============\n')
        #     writer.write(' '.join([args.index_2_word[int(j)] for j in i]))
        #     writer.write('\n************\n\n')
        #     k = k + 1
    writer.close()


def train_vae(train_iter, eval_iter, model, args):
    save_dir = "../model/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    tensor    = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    cur_epoch = 0
    for epoch in range(args.num_epoch):
        model.train()
        tracker = defaultdict(tensor)
        iteration = 0
        for batch in train_iter:    
            
            # Forward pass
            sample     = batch.text[0]
            length     = batch.text[1]
            length     = torch.add(length, -1)
            batch_size = len(sample)
            feature    = Variable(sample)
            _input     = feature[:, :-1]
            target     = feature[:, 1:]
            logp, mean, logv, z = model(_input, length, _input)
            
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
                length, mean, logv, args.anneal_function, step, args.k, args.x0, model.pad_idx)

            loss = (NLL_loss + KL_weight * KL_loss)/batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.unsqueeze(0)))
            if iteration % args.print_every == 0:
                print("Train: Batch %04d, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                %(iteration, loss.data[0], NLL_loss.data[0]/batch_size, KL_loss.data[0]/batch_size, KL_weight))
            if step % 200 == 0:
                eval_vae(model, eval_iter, args, step, cur_epoch, iteration)
                model.train()
            iteration += 1
        cur_epoch += 1


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0, pad_idx):
    NLL      = torch.nn.NLLLoss(size_average = False, ignore_index=pad_idx)
    target   = target[:, :torch.max(length).data[0]].contiguous().view(-1)
    # logp     = logp.view(-1, logp.size(2))
    batch_size = logp.size(0)
    logp     = logp.view(-1, logp.size(2))[:batch_size*torch.max(length), :]
    NLL_loss = NLL(logp, target)
    # KL Divergence
    KL_loss   = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight









