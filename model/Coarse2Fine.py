import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pdb
from torch.autograd import Variable
from module.SegGraph import Hypergraph
from module.locked_dropout import LockedDropout
import numpy as np



class SemiEnt(nn.Module):
    """
    Only dynamic Embedding, Self RNN, Hypergraph
    """
    def __init__(self, config):
        super(SemiEnt, self).__init__()
        self.config = config

        self.rnn = nn.LSTM(config.token_embed, config.f_hidden_size, batch_first = True, 
            num_layers = config.f_layers, dropout = config.f_lstm_dropout, bidirectional = True)
        self.char_rnn = nn.LSTM(config.char_embed, config.c_hidden_size, batch_first = True, 
            num_layers = config.c_layers, dropout = config.c_lstm_dropout, bidirectional = True)

        self.word_embed = nn.Embedding(config.voc_size, config.token_embed)
        self.char_embed = nn.Embedding(config.char_voc_size, config.char_embed)
        # self.input_dropout = nn.Dropout(config.input_dropout)
        self.input_dropout = LockedDropout(config.input_dropout)

        self.hypergraph = Hypergraph(config)

        self.disc_rnn = nn.LSTM(config.f_hidden_size * 2, config.d_hidden_size, batch_first = True,
            num_layers = 1, bidirectional = True )
        self.bin_class_feat = nn.Linear(config.d_hidden_size * 2, 2)
        self.d_dropout = nn.Dropout(config.d_dropout)

        self.class_weight = torch.FloatTensor([1, self.config.class_weight])
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.class_weight)
    
    
    def run_char_lstm(self, char_seqs):
        ret = []
        for char_seq in char_seqs:
            char_vec = self.char_embed(char_seq)
            char_vec = char_vec.unsqueeze(0)
            lstm_out, (hid_states, cell_states) = self.char_rnn(char_vec)
            word_rep = torch.cat([hid_states[0,0,:], hid_states[1,0,:]], 0) # 2 * c_hidden_size
            ret.append(word_rep)
        ret_vec = torch.stack(ret, 0)
        return ret_vec


    def forward_coarse(self, token_vec, char_vecs, entities):
        word_vec = self.word_embed(token_vec)
        # char_vec = self.run_char_lstm(char_vecs)
        # word_vec = torch.cat([word_vec, char_vec], 1)
        word_vec = self.input_dropout(word_vec)

        word_vec = word_vec.unsqueeze(0)
        lstm_out, (hid_states, cell_states) = self.rnn(word_vec)
        feat2hyper = lstm_out

        segments = self.disco2coarse(entities)
        label_batch = [segments] # batch size 1
        ret_dic = self.hypergraph(feat2hyper, label_batch)
        return ret_dic["loss"]


    def predict_coarse(self, token_vec, char_vecs):
        word_vec = self.word_embed(token_vec)
        # char_vec = self.run_char_lstm(char_vecs)
        # word_vec = torch.cat([word_vec, char_vec], 1)

        word_vec = word_vec.unsqueeze(0)
        lstm_out, (hid_states, cell_states) = self.rnn(word_vec)
        feat2hyper = lstm_out
        return self.hypergraph.decode(feat2hyper)[0]  # fake batch

    
    def forward_fine(self, sample_vars):
        (no_hole_x1, no_hole_x2), no_hole_y = sample_vars[0]
        (one_hole_x1, one_hole_x2, one_hole_x3, one_hole_x4), one_hole_y = sample_vars[1]
        if sample_vars[2] is not None:
            (two_hole_x1, two_hole_x2, two_hole_x3, two_hole_x4, two_hole_x5, two_hole_x6), two_hole_y = sample_vars[2]

        span_mat = self.hypergraph.span_cache

        no_hole_seg = span_mat[no_hole_x1, no_hole_x2, 0, :]
        no_hole_seg = no_hole_seg.unsqueeze(1)
        no_hole_rnn, (ht, ct) = self.disc_rnn(no_hole_seg)
        
        class_feat = F.relu(torch.cat([ht[0], ht[1]], 1))
        no_hole_feat = self.bin_class_feat(self.d_dropout(class_feat))
        no_hole_loss = self.cross_entropy(no_hole_feat, no_hole_y)

        one_hole_seg = [ span_mat[one_hole_x1, one_hole_x2, 0, :], span_mat[one_hole_x3, one_hole_x4, 0, :] ]
        one_hole_seg = torch.stack(one_hole_seg, 1)
        one_hole_rnn, (ht, ct) = self.disc_rnn(one_hole_seg)
        class_feat = F.relu(torch.cat([ht[0], ht[1]], 1))
        one_hole_feat = self.bin_class_feat(self.d_dropout(class_feat))
        one_hole_loss = self.cross_entropy(one_hole_feat, one_hole_y)

        if sample_vars[2] is not None:
            two_hole_seg = [ span_mat[two_hole_x1, two_hole_x2, 0, :], span_mat[two_hole_x3, two_hole_x4, 0, :], 
                span_mat[two_hole_x5, two_hole_x6, 0, :] ]
            two_hole_seg = torch.stack(two_hole_seg, 1)
            two_hole_rnn, (ht, ct) = self.disc_rnn(two_hole_seg)
            class_feat = F.relu(torch.cat([ht[0], ht[1]], 1))
            two_hole_feat = self.bin_class_feat(self.d_dropout(class_feat))
            two_hole_loss = self.cross_entropy(two_hole_feat, two_hole_y)

            all_loss = no_hole_loss + one_hole_loss + two_hole_loss
        else:
            all_loss = no_hole_loss + one_hole_loss
        
        return all_loss


    def predict_fine(self, segments):
        pred_ents = []
        (no_hole_x_var, no_hole_x), (one_hole_x_var, one_hole_x), (two_hole_x_var, 
            two_hole_x) = self.gen_samples(segments, self.hypergraph.span_cache)

        if len(no_hole_x) > 0:
            no_hole_rnn, (ht, ct) = self.disc_rnn(no_hole_x_var)
            class_feat = F.relu(torch.cat([ht[0], ht[1]], 1))
            no_hole_feat = self.bin_class_feat(class_feat)
            no_hole_labels = torch.max(no_hole_feat, 1)[1].cpu().numpy()
            for i_, l_ in enumerate(no_hole_labels):
                if l_ == 1:  pred_ents.append(no_hole_x[i_])
        
        if len(one_hole_x) > 0:
            one_hole_rnn, (ht, ct) = self.disc_rnn(one_hole_x_var)
            class_feat = F.relu(torch.cat([ht[0], ht[1]], 1))
            one_hole_feat = self.bin_class_feat(class_feat)
            one_hole_labels = torch.max(one_hole_feat, 1)[1].cpu().numpy() 
            for i_, l_ in enumerate(one_hole_labels):
                if l_ == 1:  pred_ents.append(one_hole_x[i_])
        
        if len(two_hole_x) > 0:
            two_hole_rnn, (ht, ct) = self.disc_rnn(two_hole_x_var)
            class_feat = F.relu(torch.cat([ht[0], ht[1]], 1))
            two_hole_feat = self.bin_class_feat(class_feat)
            two_hole_labels = torch.max(two_hole_feat, 1)[1].cpu().numpy() 
            for i_, l_ in enumerate(two_hole_labels):
                if l_ == 1:  pred_ents.append(two_hole_x[i_])

        # pdb.set_trace()
        return pred_ents
    
    
    def forward(self, token_vec, char_vecs, entities, sample_vars):
        coarse_loss = self.forward_coarse(token_vec, char_vecs, entities)
        fine_loss = self.forward_fine(sample_vars)
        return coarse_loss + fine_loss


    def predict(self, token_vec, char_vecs):
        segs = self.predict_coarse(token_vec, char_vecs)
        ents = self.predict_fine(segs)
        return ents


    def load_vector(self, word2vec):
        t_v = torch.Tensor(word2vec)
        print("Loading embedding size size {}".format( t_v.size()))
        self.word_embed.weight = nn.Parameter(t_v)
        # self.word_embed.weight.requires_grad = False


    def disco2coarse(self, entities):
        """
        Extract all the segments
        """
        segments = set()
        for entity in entities:
            for seg in entity:
                segments.add((seg[0], seg[1], 0)) # default label
        
        segments = list(segments)
        return segments

    
    def gen_samples(self, segs, span_mat):
        """
        generate all the possible disco entities
        """
        segments = list(segs)
        segments.sort(key = lambda x: x[0]) #descending order
        
        no_hole_x, one_hole_x, two_hole_x = [], [], []
        no_hole_x_var = []
        if len(segments) > 0:
            for seg in segments:
                no_hole_x.append([(seg[0], seg[1])])
                no_hole_x_var.append( span_mat[seg[0], seg[1], 0, :] )
            no_hole_x_var = torch.stack(no_hole_x_var, 0).unsqueeze(1) # ent_num * 1 * feat_dim
            if self.config.if_gpu:
                no_hole_x_var = no_hole_x_var.cuda()
        
        one_hole_x1_var = []
        one_hole_x2_var = []
        one_hole_x_var = None
        if len(segments) > 1:
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    first = segments[i]
                    second = segments[j]

                    if second[0] - first[1] <= 17 and second[0] > first[1] + 1:
                        ent = [(first[0], first[1]), (second[0], second[1])]
                        one_hole_x.append(ent)
                        one_hole_x1_var.append(span_mat[first[0], first[1], 0, :])
                        one_hole_x2_var.append(span_mat[second[0], second[1], 0, :])
            if len(one_hole_x) > 0:
                one_hole_x1_var = torch.stack(one_hole_x1_var, 0)
                one_hole_x2_var = torch.stack(one_hole_x2_var, 0)
                one_hole_x_var = torch.stack([one_hole_x1_var, one_hole_x2_var], 1) # ent_num * 2 * feat_dum
                if self.config.if_gpu:
                    one_hole_x_var = one_hole_x_var.cuda()
        
        two_hole_x1_var = []
        two_hole_x2_var = []
        two_hole_x3_var = []
        two_hole_x_var = None
        if len(segments) > 2:
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    for k in range(j + 1, len(segments)):
                        first = segments[i]
                        second = segments[j]
                        third = segments[k]

                        if third[0] - second[1] <= 3 and third[0] > second[1] + 1 and \
                            second[0] - first[1] <= 3 and second[0] > first[1] + 1:
                            ent = [(first[0], first[1]), (second[0], second[1]), (third[0], third[1])]
                            two_hole_x.append(ent)
                            two_hole_x1_var.append(span_mat[first[0], first[1], 0, :])
                            two_hole_x2_var.append(span_mat[second[0], second[1], 0, :])
                            two_hole_x3_var.append(span_mat[third[0], third[1], 0, :])
            if len(two_hole_x) > 0:
                two_hole_x1_var = torch.stack(two_hole_x1_var, 0)
                two_hole_x2_var = torch.stack(two_hole_x2_var, 0)
                two_hole_x3_var = torch.stack(two_hole_x3_var, 0)
                two_hole_x_var = torch.stack([two_hole_x1_var, two_hole_x2_var, two_hole_x3_var], 1)
                if self.config.if_gpu:
                    two_hole_x_var = two_hole_x_var.cuda()

        return (no_hole_x_var, no_hole_x), (one_hole_x_var, one_hole_x), (two_hole_x_var, two_hole_x)
