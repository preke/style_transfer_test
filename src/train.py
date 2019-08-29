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
import adabound

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pickle



from collections import Counter
import math

import subprocess
import dataload
from model import *
from utils import preprocess_write, get_pretrained_word_embed, preprocess_pos_neg
from dataload import load_data, load_pos_neg_data
from collections import OrderedDict, defaultdict

# logging
# import logging
# import logging.config
# config_file = 'logging.ini'
# logging.config.fileConfig(config_file, disable_existing_loggers=False)
# logger = logging.getLogger(__name__)



def eval_vae(model, eval_iter, args, step, cur_epoch, iteration, sentiment_classifier, w2v_model,
    Best_acc, Best_BLEU, Best_WMD, temp):
    '''
    currently: test.mask, should be: reference.mask
    Evaluation should transfer the sentiment;
    Should be 
    '''

    model.eval()
    
    Total_loss           = torch.tensor(0.0).cuda()
    Total_NLL_loss       = torch.tensor(0.0).cuda()
    Total_KL_loss        = torch.tensor(0.0).cuda()
    Total_sentiment_loss = torch.tensor(0.0).cuda()
    
    cnt            = 0
    senti_corrects = 0
    writer         = open('res/yelp_vae_epoch_'+str(cur_epoch) + '_batch_' + str(iteration) + '_.txt', 'w')
    val_bleu = AverageMeter()
    val_wmd  = AverageMeter()

    for batch in eval_iter:
        cnt += 1    
        mask_sample         = batch.mask_text[0]
        mask_feature        = Variable(mask_sample)
        length              = batch.mask_text[1]
        length              = torch.add(length, -1)
        mask_input          = mask_feature[:, :-1]
        batch_size          = len(mask_sample)
        target              = batch.text[0]
        label               = batch.label
        
        logp, mean, logv, z = model(mask_input, length, mask_input, False)
        NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
            length, mean, logv, args.anneal_function, step, args.k, args.x0, model.pad_idx)

        # logp = torch.argmax(logp, dim=2)
        logp = gumbel_softmax_sample(logp, temp)
        arg_max_logp = torch.argmax(logp, dim=2)

        k = 0 
        pred_list = []
        target_list = []
        for i in arg_max_logp:
            mask_sample_ = [args.index_2_word[int(l)] for l in mask_sample[k]]
            target_ = [args.index_2_word[int(l)] for l in target[k]]
            pred    = [args.index_2_word[int(j)] for j in i]
            pred_list.append(pred)
            target_list.append(target_)
            writer.write(' '.join(mask_sample_))
            writer.write('\n.................\n')
            writer.write(' '.join(pred))
            writer.write('\n=============\n')
            writer.write(' '.join(target_))
            writer.write('\n************\n\n')
            k = k + 1
            val_wmd.update(w2v_model.wmdistance(pred, target_), 1)
        
        bleu_value = get_bleu(pred_list, target_list)
        val_bleu.update(bleu_value, 1)

        sentiment      = sentiment_classifier(arg_max_logp)
        sentiment_loss = F.cross_entropy(sentiment, label)
        loss           = (NLL_loss + KL_weight * KL_loss + sentiment_loss)/batch_size
        
        Total_loss     += float(loss)
        Total_NLL_loss += float(NLL_loss)/batch_size
        Total_KL_loss  += float(KL_loss)/batch_size
        Total_sentiment_loss  += float(sentiment_loss)/batch_size
        
        # convert the label of transfered sentences
        senti_corrects += (torch.max(sentiment, 1)
                     [1].view(label.size()).data == label.data).sum()

        del loss
        del sentiment_loss
        del NLL_loss
        del KL_loss
        del KL_weight
        del arg_max_logp
        del mean
        del logv
        del z
    writer.close()
    
    print('AVG Evaluation BLEU_score is:%s\n'%(str(val_bleu.avg)))
    print('AVG Evaluation WMD_score is:%s\n'%(str(val_wmd.avg)))
    print("Valid: Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, Senti-Loss %9.4f"
                %(Total_loss.data[0]/cnt, Total_NLL_loss.data[0]/cnt, Total_KL_loss.data[0]/cnt, Total_sentiment_loss.data[0]/cnt))
    
    size = len(eval_iter.dataset)
    # accuracy = float(100.0 * (1.0 - float(senti_corrects)/size))
    accuracy = float(100.0 * (float(senti_corrects)/size))
    print('Evaluation acc: {:.4f}%({}/{}) \n'.format(accuracy, senti_corrects, size))

    if accuracy > Best_acc or val_bleu.avg > Best_BLEU or val_wmd.avg < Best_WMD: 
        Best_acc  = accuracy
        Best_BLEU = val_bleu.avg
        Best_WMD  = val_wmd.avg
        if accuracy >= 0.0: 
            save_path = 'saved_model/yelp_acc_'+str(accuracy)+'_bleu_'+str(val_bleu.avg)+'_wmd_'+str(val_wmd.avg)+'_.pt'
            torch.save(model.state_dict(), save_path)
            print('Save model to ' + save_path)




def test_vae(model, test_iter, args, sentiment_classifier, w2v_model):
    '''
        Transfer the sentiment to show the performance
        HUMAN REFERENCE
    '''
    model.eval()
    
    cnt            = 0
    senti_corrects = 0
    writer         = open('res/test_yelp_vae_.txt', 'w')
    test_bleu = AverageMeter()
    test_wmd  = AverageMeter()
    for batch in test_iter:
        cnt += 1    
        mask_sample         = batch.mask_text[0]
        mask_feature        = Variable(mask_sample)
        length              = batch.mask_text[1]
        length              = torch.add(length, -1)
        mask_input          = mask_feature[:, :-1]
        batch_size          = len(mask_sample)
        target              = batch.target[0]
        label               = batch.label
        
        logp, mean, logv, z = model(mask_input, length, mask_input, True)
        arg_max_logp = torch.argmax(logp, dim=2)
        # logp = gumbel_softmax_sample(logp, temp)
        # arg_max_logp = torch.argmax(logp, dim=2)

        k = 0 
        pred_list = []
        target_list = []
        for i in arg_max_logp:
            mask_sample_ = [args.index_2_word[int(l)] for l in mask_sample[k]]
            target_ = [args.index_2_word[int(l)] for l in target[k]]
            pred    = [args.index_2_word[int(j)] for j in i]
            pred_list.append(pred)
            target_list.append(target_)
            writer.write(' '.join(mask_sample_))
            writer.write('\n.................\n')
            writer.write(' '.join(pred))
            writer.write('\n=============\n')
            writer.write(' '.join(target_))
            writer.write('\n************\n\n')
            k = k + 1
            test_wmd.update(w2v_model.wmdistance(pred, target_), 1)
        
        bleu_value = get_bleu(pred_list, target_list)
        test_bleu.update(bleu_value, 1)

        sentiment      = sentiment_classifier(arg_max_logp)
        sentiment_loss = F.cross_entropy(sentiment, label)
        loss           = (NLL_loss + KL_weight * KL_loss + sentiment_loss)/batch_size
        
        Total_loss           += float(loss)
        Total_NLL_loss       += float(NLL_loss)/batch_size
        Total_KL_loss        += float(KL_loss)/batch_size
        Total_sentiment_loss += float(sentiment_loss)/batch_size
        
        # convert the label of transfered sentences
        senti_corrects += (torch.max(sentiment, 1)
                     [1].view(label.size()).data == label.data).sum()

    writer.close()    
    print('AVG TEST BLEU_score is:%s\n'%(str(test_bleu.avg)))
    print('AVG TEST WMD_score is:%s\n'%(str(test_wmd.avg)))
    print("Valid: Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, Senti-Loss %9.4f"
                %(Total_loss.data[0]/cnt, Total_NLL_loss.data[0]/cnt, Total_KL_loss.data[0]/cnt, Total_sentiment_loss.data[0]/cnt))
    
    size = len(test_iter.dataset)
    # accuracy = float(100.0 * (1.0 - float(senti_corrects)/size))
    accuracy = float(100.0 * (float(senti_corrects)/size))
    print('TEST acc: {:.4f}%({}/{}) \n'.format(accuracy, senti_corrects, size))







def train_vae(train_iter, eval_iter, model, args, sentiment_classifier):
    save_dir = "../model/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = adabound.AdaBound(model.parameters, lr=args.lr, final_lr=0.1)
    tensor    = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step      = 0
    cur_epoch = 0
    Best_acc  = 0.0
    Best_BLEU = 0.0
    Best_WMD  = 100.0

    # gumbel softmax
    temp        = 1.0
    temp_min    = 0.1
    ANNEAL_RATE = 0.00003
    

    w2v_model = Word2Vec.load("yelp_word2vec.model")
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
            label        = batch.label
            mask_sample  = batch.mask_text[0]
            mask_feature = Variable(mask_sample)
            mask_input   = mask_feature[:, :-1]
            
            logp, mean, logv, z = model(mask_input, length, mask_input)
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, target,
                length, mean, logv, args.anneal_function, step, args.k, args.x0, model.pad_idx)

            
            if iteration % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * iteration), temp_min)
            
            logp = gumbel_softmax_sample(logp, temp)
            argmax_logp = torch.argmax(logp, dim=2)
            # nearly one-hot vector to choose words
            # one_hot_logp = to_one_hot(logp)
            # logp = logp + (arg_max_logp - logp)
            
            sentiment      = sentiment_classifier(argmax_logp)
            sentiment_loss = F.cross_entropy(sentiment, label)
            loss           = (NLL_loss + KL_weight * KL_loss + sentiment_loss)/batch_size
            # loss           = (NLL_loss + KL_weight * KL_loss)/batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.unsqueeze(0)))
            if iteration % args.print_every == 0:
                print("Train: Batch %04d, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, Senti-Loss: %9.4f"
                %(iteration, loss.data[0], NLL_loss.data[0]/batch_size, KL_loss.data[0]/batch_size, KL_weight, sentiment_loss.data[0]/batch_size))

                
            if step % 500 == 0 and step > 0:
                eval_vae(model, eval_iter, args, step, cur_epoch, iteration, sentiment_classifier, w2v_model,
                    Best_acc, Best_BLEU, Best_WMD, temp)

                model.train()
            iteration += 1
        cur_epoch += 1


# def to_one_hot(logp):
#     tmp0     = logp.clone() # b * s * v
#     max_logp = torch.argmax(logp, dim=2) # b * s




def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, args):
    hard = False
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    categorical_dim = args.vocab_size
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y.view(-1, categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)



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
    # optimizer = torch.optim.Adam(parameters, lr=args.lr)
    # optimizer = torch.optim.SGD(parameters, lr=args.lr)
    optimizer = adabound.AdaBound(parameters, lr=args.lr, final_lr=0.1)
    steps = 0
    best_acc = 0
    best_loss = 10000
    last_step = 0
    loss_list = []
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
            if steps % args.test_interval == 0 and steps != 0:
                acc, avg_loss = eval_cnn(dev_iter, model, args)
                loss_list.append(avg_loss)
                if avg_loss < best_loss:
                    best_loss = avg_loss
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
    accuracy = 100.0 * float(corrects)/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy, avg_loss


def save_cnn(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)









def model_perplexity(model, src, src_test, trg, trg_test, config, loss_criterion, src_valid=None, trg_valid=None, verbose=False):
    """Compute model perplexity."""
    # Get source minibatch
    losses = []
    for j in xrange(0, len(src_test['data']) // 100, config['data']['batch_size']):
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )
        input_lines_src = Variable(input_lines_src.data, volatile=True)
        output_lines_src = Variable(input_lines_src.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        # Get target minibatch
        input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
            get_minibatch(
                trg_test['data'], trg['word2id'], j,
                config['data']['batch_size'], config['data']['max_trg_length'],
                add_start=True, add_end=True
            )
        )
        input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
        output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        decoder_logit = model(input_lines_src, input_lines_trg_gold)

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
            output_lines_trg_gold.view(-1)
        )

        losses.append(loss.data[0])

    return np.exp(np.mean(losses))



def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)



def tensor2np(tensor):
    return tensor.data.cpu().numpy()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Borrowed from ImageNet training in PyTorch project
    https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def randomChoice(batch_size):
    return random.randint(0, batch_size - 1)







