import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='yoochoose1_64/diginetica/Tmall/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--theta', type=int, default=2)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))


def preprocess(data):
    # Default Padding item
    PAD_Token = 0
    item2idx = {}
    idx2item = {PAD_Token: 'PAD'}
    num_items = 1
    max_length = 0

    for seq in data:
        if len(seq) > max_length:
            max_length = len(seq)
        for item in seq:
            if item not in item2idx:
                item2idx[item] = num_items
                idx2item[num_items] = item
                num_items += 1
    print(max(item2idx.keys()))
    # transfer to idx
    data = transfer_item2idx(data, item2idx)
    return data, num_items, max_length, item2idx, idx2item


def transfer_item2idx(data, item2idx):
    for i in range(len(data)):
        data[i] = [item2idx[s] for s in data[i]]
    return data


seq, num, max_length, item2idx, idx2item = preprocess(seq)
print(num)

# if dataset == 'diginetica':
#     num = 43098
# elif dataset == "Tmall":
#     num = 40728
# elif dataset == "Nowplaying":
#     num = 60417
# else:
#     num = 3

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 2 * opt.theta):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(num)]

neig_num = 0
for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]
    neig_num += len(adj[t])
print('mean neighbors: ', neig_num / num)

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

# 存一个邻接矩阵
adj_numpy = np.zeros((num, num), dtype=np.int)
for i in range(1, num):
    for idx, val in adj1[i].items():
        adj_numpy[i][idx] = val
ratio = 1 - len(np.nonzero(adj_numpy)) / (num * num)
print('稀疏率：%0.6f %%' % (100. * ratio))

# 对weight进行归一化到[0, 1]
for i in range(1, num):
    if len(weight[i]) != 0:
        weight[i] = (np.asarray(weight[i]) / np.sum(weight[i])).tolist()

param = {
    'num_items': num,
    'max_length': max_length,
    'item2idx': item2idx,
    'idx2item': idx2item
}

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(adj_numpy, open('datasets/' + dataset + '/adj_matrix' + '.pkl', 'wb'), protocol=4)
pickle.dump(param, open('datasets/' + dataset + '/parm.pkl', 'wb'))

