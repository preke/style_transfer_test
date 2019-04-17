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

def eval_vae(model, eval_iter, args, step, cur_epoch, iteration):
    model.eval()
    
    Total_loss     = torch.tensor(0.0).cuda()
    Total_NLL_loss = torch.tensor(0.0).cuda()
    Total_KL_loss  = torch.tensor(0.0).cuda()
    cnt            = 0
    writer         = open('res/vae_epoch_'+str(cur_epoch) + '_batch_' + str(iteration) + '_.txt', 'w')
    for batch in eval_iter:
        cnt += 1
        sample     = batch.text[0]
        length     = batch.text[1]
        length     = torch.add(length, -1)
        batch_size = len(sample)
        feature    = Variable(sample)
        _input     = feature[:, :-1]
        target     = feature[:, 1:]

        mask_sample  = batch.mask_text[0]
        mask_feature = Variable(mask_sample)
        mask_input   = mask_feature[:, :-1]
        
        logp, mean, logv, z = model(_input, length, mask_input)
        
        
        NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
            length, mean, logv, args.anneal_function, step, args.k, args.x0, model.pad_idx)
        loss = (NLL_loss + KL_weight * KL_loss)/batch_size
        
        Total_loss     += loss
        Total_NLL_loss += NLL_loss/batch_size
        Total_KL_loss  += KL_loss/batch_size

        logp = torch.argmax(logp, dim=2)
        # print(generations)
        # NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
        #     length, mean, logv, args.anneal_function, step, args.k, args.x0, model.pad_idx)

        k = 0 
        for i in logp:
            writer.write(' '.join([args.index_2_word[int(l)] for l in sample[k]]))
            writer.write('\n=============\n')
            writer.write(' '.join([args.index_2_word[int(j)] for j in i]))
            writer.write('\n************\n\n')
            k = k + 1
    writer.close()
    logger.info('\n')
    
    print("Valid: Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f"
                %(Total_loss.data[0]/cnt, Total_NLL_loss.data[0]/cnt, Total_KL_loss.data[0]/cnt))
    
    # logger.info('\n')
    save_path = 'saved_model/epoch_'+str(cur_epoch) + '_batch_' + str(iteration) + '_.pt'
    torch.save(model.state_dict(), save_path)
    logger.info('Save model to ' + save_path)


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
            sample       = batch.text[0]
            length       = batch.text[1]
            length       = torch.add(length, -1)
            batch_size   = len(sample)
            feature      = Variable(sample)
            _input       = feature[:, :-1]
            target       = feature[:, 1:]

            mask_sample  = batch.mask_text[0]
            mask_feature = Variable(mask_sample)
            mask_input   = mask_feature[:, :-1]
            
            logp, mean, logv, z = model(mask_input, length, _input)
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
            if step % 2000 == 0:
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




def train_cnn(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text[0], batch.label
            # feature.data.t_(), target.data.sub_(1)  # batch first, index align
            # if args.cuda:
            #     feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval_cnn(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save_cnn(model, args.cnn_save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.cnn_save_dir, 'snapshot', steps)


def eval_cnn(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text[0], batch.label
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        # if args.cuda:
        #     feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def save_cnn(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
























