
import torch.nn as nn
import torch
import numpy as np
from aggregator import *


class NeighRoutingGnnCls2Scores(nn.Module):
    def __init__(self, opt, n_items, adj_items, weight_items, prob, device):
        super(NeighRoutingGnnCls2Scores, self).__init__()
        self.n_items = n_items
        self.hidden_size = opt.hidden_size
        self.routing_iter = opt.routing_iter
        self.K = opt.n_factors
        self.device = device

        self.adj_items = torch.LongTensor(adj_items).to(device)
        self.weight_items = torch.FloatTensor(weight_items).to(device)
        self.prob = torch.FloatTensor(prob).to(device)
        self.prob_emb = nn.Parameter(self.prob, requires_grad=True)

        self.hop = opt.hop
        self.sample_num = opt.sample_num

        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=1, batch_first=True, bidirectional=False)

        # Global Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = NeighborRoutingAgg(self.hidden_size, self.routing_iter, self.device)
            self.add_module('agg_gnn_{}'.format(i), agg)
            self.global_agg.append(agg)

        self.cls_embeddings = nn.Linear(self.K, self.hidden_size, bias=False)

        self.a1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.dropout = nn.Dropout(0.2)
        self.loss_function1 = nn.CrossEntropyLoss()
        self.loss_function2 = nn.CrossEntropyLoss()
        self.loss_function = nn.CrossEntropyLoss()

        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.LN3 = nn.LayerNorm(self.hidden_size)
        self.LN4 = nn.LayerNorm(self.hidden_size)

    def init_h0(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, inp_sess, mask_1, mask_inf, lengths):
        batch_size = inp_sess.size(0)
        seqs_len = inp_sess.size(1)

        x = self.item_embeddings.weight[1:]
        x_nb = self.adj_items[1:]

        out_vectors = [x]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(x=out_vectors[i], x_nb=x_nb)
            out_vectors.append(x)

        item_vectors = out_vectors[0]
        for i in range(1, len(out_vectors)):
            item_vectors = item_vectors + out_vectors[i]
        item_vectors = self.LN1(item_vectors)

        # pad 0 item
        pad_vector = torch.zeros(1, self.hidden_size).to(self.device)
        item_vectors = torch.cat((pad_vector, item_vectors), dim=0)

        inp_emb = self.dropout(item_vectors[inp_sess])     # bs * L * d
        h0 = self.init_h0(batch_size)
        H, _ = self.gru(inp_emb, h0)
        H = self.LN2(H)
        ht = H[torch.arange(H.size(0)), lengths - 1]

        inp_cls = self.prob_emb[inp_sess]  # bs * L * K
        # Transfer to cls embeddings
        # inp_cls_emb = self.cls_embeddings(inp_cls)
        inp_cls_emb = self.LN3(self.cls_embeddings(inp_cls))

        H_cls, _ = self.gru(inp_cls_emb)
        H_cls = self.LN4(H_cls)
        ht_cls = H_cls[torch.arange(batch_size), lengths - 1]

        scores1 = torch.matmul(ht, item_vectors[1:].transpose(1, 0))
        # scores2 = torch.matmul(ht_cls, self.cls_embeddings(self.prob_emb[1:]).transpose(1, 0))
        scores2 = torch.matmul(ht_cls, self.LN3(self.cls_embeddings(self.prob_emb[1:])).transpose(1, 0))

        scores = F.sigmoid(self.a1) * scores1 + F.sigmoid(self.a2) * scores2
        return scores, scores1, scores2
