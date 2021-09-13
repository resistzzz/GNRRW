
import numpy as np


def transfer2idx(data, item2idx):
    seqs, labs = data[0], data[1]
    for i in range(len(seqs)):
        data[0][i] = [item2idx[s] for s in data[0][i]]
        data[1][i] = item2idx[data[1][i]]
    return data


def handle_adj(adj_items, weight_items, n_items, sample_num):
    adj_entity = np.zeros((n_items, sample_num), dtype=np.int)
    wei_entity = np.zeros((n_items, sample_num))
    for entity in range(1, n_items):
        neighbor = list(adj_items[entity])
        neighbor_weight = list(weight_items[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        tmp, tmp_wei = [], []
        for i in sampled_indices:
            tmp.append(neighbor[i])
            tmp_wei.append(neighbor_weight[i])

        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        wei_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
    return adj_entity, wei_entity


class Data(object):
    '''Data用每个batch中的最长长度来进行padding'''
    def __init__(self, data, n_items):
        self.data = data
        self.n_items = n_items

        max_len = 0
        for seq in data[0]:
            if len(seq) > max_len:
                max_len = len(seq)
        self.max_len = max_len
        self.raw_sessions = np.asarray(data[0])
        self.raw_labs = np.asarray(data[1])
        self.length = len(self.raw_sessions)

    def __len__(self):
        return self.length

    def generate_batch(self, batch_size):
        n_batch = self.length // batch_size
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice_sess_mask(self, index):
        inp_sess = self.raw_sessions[index]
        targets = self.raw_labs[index]
        lengths = []
        for session in inp_sess:
            lengths.append(len(session))
        max_length = max(lengths)
        inp_sess, mask_1, mask_inf = self.zero_padding_mask(inp_sess, max_length)
        return inp_sess, targets, mask_1, mask_inf, lengths

    def zero_padding_mask(self, data, max_length):
        out_data = np.zeros((len(data), max_length), dtype=np.int)
        mask_1 = np.zeros((len(data), max_length), dtype=np.int)
        mask_inf = np.full((len(data), max_length), float('-inf'), dtype=np.float32)
        for i in range(len(data)):
            out_data[i, :len(data[i])] = data[i]
            mask_1[i, :len(data[i])] = 1
            mask_inf[i, :len(data[i])] = 0.0
        return out_data, mask_1, mask_inf
