import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

from utils import *

# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.enc_h_in = nn.Linear(hidden_dim*2, hidden_dim)
        self.prev_s_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, enc_h, prev_s):
        '''
        enc_h  : B x S x H 
        prev_s : B x 1 x H 
        '''
        seq_len = enc_h.size(1) 
        enc_h_in = self.enc_h_in(enc_h) # B x S x H
        prev_s = self.prev_s_in(prev_s).unsqueeze(1)  # B x 1 x H

        h = F.tanh(enc_h_in + prev_s.expand_as(enc_h_in)) # B x S x H
        h = self.linear(h)  # B x S x 1

        alpha = F.softmax(h)
        ctx = torch.bmm(alpha.transpose(2,1), enc_h).squeeze(1) # B x 1 x H

        return ctx 

class DecoderCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(DecoderCell, self).__init__()

        self.input_weights  = nn.Linear(embed_dim, hidden_dim*2)
        self.hidden_weights = nn.Linear(hidden_dim, hidden_dim*2)
        self.ctx_weights    = nn.Linear(hidden_dim*2, hidden_dim*2)
        
        self.input_in  = nn.Linear(embed_dim, hidden_dim)
        self.hidden_in = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_in    = nn.Linear(hidden_dim*2, hidden_dim)

        

    def forward(self, trg_word, prev_s, ctx):        
        gates = self.input_weights(trg_word) + self.hidden_weights(prev_s) + self.ctx_weights(ctx)
        reset_gate, update_gate = gates.chunk(2, 1)
        reset_gate = F.sigmoid(reset_gate)
        update_gate = F.sigmoid(update_gate)
        prev_s_tilde = self.input_in(trg_word) + self.hidden_in(prev_s) + self.ctx_in(ctx)
        prev_s_tilde = F.tanh(prev_s_tilde)

        # print reset_gate.size()
        prev_s = torch.mul((1-reset_gate), prev_s) + torch.mul(reset_gate, prev_s_tilde)
        return prev_s

class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.vocab_size
        D = args.embed_dim
        
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        # use pre-trained
        # if args.word_Embedding:
            # pass
        self.embed.weight.data.copy_(args.pretrained_weight)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(300, 100)
    
    def forward(self, q1):
        q1 = self.embed(q1)
        q1 = q1.unsqueeze(1)  # (N, Ci, W, D)
        q1 = [F.tanh(conv(q1)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        q1 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q1]  # [(N, Co), ...]*len(Ks)
        q1 = [F.tanh(i) for i in q1]
        q1 = torch.cat(q1, 1) # 32 * 300
        q1 = self.fc1(q1)
        return q1

        
        



class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len, trg_soi, pre_embedding):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.trg_soi = trg_soi
        
        self.embed = nn.Embedding(vocab_size, embed_dim) 
        self.embed.weight.data.copy_(pre_embedding)       
        self.attention = Attention(hidden_dim) 
        self.decodercell = DecoderCell(embed_dim, hidden_dim)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)


    def forward(self, enc_h, prev_s, target=None):
        '''
        enc_h  : B x S x 2*H 
        prev_s : B x H
        '''

        if target is not None:
            batch_size, target_len = target.size(0), target.size(1)
            
            dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))

            if torch.cuda.is_available():
                dec_h = dec_h.cuda()

            target = self.embed(target)  
            for i in range(target_len):
                ctx = self.attention(enc_h, prev_s)                     
                prev_s = self.decodercell(target[:, i], prev_s, ctx)
                dec_h[:,i,:] = prev_s

            outputs = self.dec2word(dec_h)


        else:
            batch_size = enc_h.size(0)
            target = Variable(torch.LongTensor([self.trg_soi] * batch_size), volatile=True).view(batch_size, 1)
            outputs = Variable(torch.zeros(batch_size, self.max_len, self.vocab_size))
            
            
            if torch.cuda.is_available():
                target = target.cuda()
                outputs = outputs.cuda()
            
            for i in range(1, self.max_len):
                target = self.embed(target).squeeze(1)              
                ctx = self.attention(enc_h, prev_s)                 
                prev_s = self.decodercell(target, prev_s, ctx)
                output = self.dec2word(prev_s)
                outputs[:,i,:] = output
                target = output.topk(1)[1]
            
        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi, args):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword

        self.encoder = Encoder(src_nword, embed_dim, hidden_dim, args)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim, max_len, trg_soi, args.pretrained_weight)

    
    def forward(self, source, src_length=None, target=None):
        batch_size = source.size(0)

        enc_h, enc_h_t = self.encoder(source, src_length) # B x S x 2*H / 2 x B x H 
        
        dec_h0 = enc_h_t[-1] # B x H 
        dec_h0 = F.tanh(self.linear(dec_h0)) # B x 1 x 2*H

        out = self.decoder(enc_h, dec_h0, target) # B x S x H
        out = F.log_softmax(out.contiguous().view(-1, self.trg_nword))
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, args):
        super(Encoder, self).__init__()
        self.num_layers = 2
        self.hidden_dim = hidden_dim
        print(args.embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(args.pretrained_weight)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def forward(self, source, src_length=None, hidden=None):
        '''
        source: B x T 
        '''
        batch_size = source.size(0)
        src_embed = self.embedding(source)
        
        if hidden is None:
            h_size = (self.num_layers *2, batch_size, self.hidden_dim)
            enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

        if src_length is not None:
            src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_length, batch_first=True)

        enc_h, enc_h_t = self.gru(src_embed, enc_h_0) 

        if src_length is not None:
            enc_h, _ = nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

        return enc_h, enc_h_t



    
