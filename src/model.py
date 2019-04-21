import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from utils import *
import time
import os
import sys

# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)



class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, pre_embedding, args, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.max_sequence_length = max_sequence_length
        self.sos_idx     = sos_idx
        self.eos_idx     = eos_idx
        self.pad_idx     = pad_idx
        self.unk_idx     = unk_idx
        self.vocab_size  = vocab_size
        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        self.embedding.weight.data.copy_(pre_embedding)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError()

        self.encoder_rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean   = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv   = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)
        self.args = args




    def encoder(self, input_sequence, sorted_lengths, batch_size):
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        return mean, logv, z

    def decoder(self, z, batch_size, sorted_idx, sorted_lengths, decoder_input, is_train):
        hidden = self.latent2hidden(z)
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        input_embedding = self.embedding(decoder_input)
        input_embedding = self.mask_to_sentiment(decoder_input, input_embedding, is_train)
        input_embedding = input_embedding.cuda()
        # decoder input
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)
        # print(outputs.sorted_indices)
        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]

        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()
        print(padded_outputs.size())
        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        return logp
        


    def mask_to_sentiment(self, input_sequence, input_embedding, is_train):
        decoder_input_embedding = []
        for i in range(input_sequence.size()[0]):
            for j in range(input_sequence.size()[1]):
                if self.args.index_2_word[input_sequence[i, j]] == 'pos':
                    if is_train:
                        input_embedding[i, j, :] = self.args.pos_rep
                    else:
                        input_embedding[i, j, :] = self.args.neg_rep
                if self.args.index_2_word[input_sequence[i, j]] == 'neg':
                    if is_train:
                        input_embedding[i, j, :] = self.args.neg_rep
                    else:
                        input_embedding[i, j, :] = self.args.pos_rep
        return input_embedding

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def forward(self, input_sequence, length, decoder_input, is_train=True):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        if decoder_input is not None:
            decoder_input = decoder_input[sorted_idx]

        # ENCODER
        mean, logv, z = self.encoder(input_sequence, sorted_lengths, batch_size)

        # DECODER
        logp = self.decoder(z, batch_size, sorted_idx, sorted_lengths, decoder_input, is_train)

        return logp, mean, logv, z

    def inference(self, z=None):

        if z is None:
            batch_size = 4
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)
        
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        # hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx     = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask    = torch.ones(batch_size, out=self.tensor()).byte()
        running_seqs     = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop
        generations      = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while(t < self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence  = input_sequence.unsqueeze(1)
            input_embedding = self.embedding(input_sequence)
            output, hidden  = self.decoder_rnn(input_embedding, hidden)

            logits         = self.outputs2vocab(output)
            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            # print(input_sequence.shape)
            # print(running_seqs)
            
            if len(running_seqs) > 0:
                if len(running_seqs) == 1 and len(input_sequence.size()) == 0:
                     input_sequence = input_sequence.unsqueeze(0)
                         
                # input_sequence = input_sequence.unsqueeze(1)
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]
                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to




    



class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V  = args.vocab_size
        D  = args.embed_dim
        C  = args.num_class
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed   = nn.Embedding(V, D)
        self.embed.weight.data.copy_(args.pretrained_weight)
        self.convs1  = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1     = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = Variable(x)

        x     = x.unsqueeze(1)  # (N, Ci, W, D)
        x     = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x     = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x     = torch.cat(x, 1)
        x     = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

