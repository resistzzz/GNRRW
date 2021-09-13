
import pickle
import numpy as np
import networkx as nx
import argparse
import time
import datetime
import os


parser = argparse.ArgumentParser(description='Mine potential topic distribution of items')
parser.add_argument('--data_path', default='data', help='dataset root path.')
parser.add_argument('--dataset', default='sample', help='dataset')
parser.add_argument('--anchor_num', type=int, default=40, help='number of potential topic for items.')
parser.add_argument('--alpha', type=float, default=0.5, help='restart factor.')
parser.add_argument('--rw', default='w', help='read or write random walk result (r or w)')
parser.add_argument('--rw_path', default='rw', help='random walk result pickle')

args = parser.parse_args()

print(args)


def build_graph(file):
    Adj_matrix = pickle.load(open(file, 'rb'))
    num_items = Adj_matrix.shape[0]
    Trans_adj = np.zeros_like(Adj_matrix, dtype=np.float)
    Adj_sum = Adj_matrix.sum(axis=1)
    for i in range(num_items):
        if Adj_sum[i] > 0:
            Trans_adj[i, :] = Adj_matrix[i, :] / Adj_sum[i]

    graph = nx.Graph()
    i_idx, j_idx = np.nonzero(Adj_matrix)
    for i in range(len(i_idx)):
        graph.add_edge(i_idx[i], j_idx[i], weight=Adj_matrix[i_idx[i], j_idx[i]])

    return Trans_adj, Adj_sum, graph


def anchor_select(graph, anchor_num):
    pagerank = nx.pagerank(graph)
    pagerank_sort = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    pagerank_sort = pagerank_sort[:anchor_num]
    anchors = [x[0] for x in pagerank_sort]

    return anchors


def random_walk(Trans_adj, anchors, alpha):
    print('start random walk...')
    anchor_num = len(anchors)
    num_items = Trans_adj.shape[0]

    # 节点分布矩阵
    prob_node = np.zeros((num_items, anchor_num))
    # 重启动矩阵
    restart = np.zeros((num_items, anchor_num))
    for i in range(anchor_num):
        restart[anchors[i]][i] = 1
        prob_node[anchors[i]][i] = 1

    count = 0
    while True:
        count += 1
        prob_t = alpha * np.dot(Trans_adj, prob_node) + (1 - alpha) * restart
        residual = np.sum(np.abs(prob_node - prob_t))
        prob_node = prob_t
        if abs(residual) < 1e-8:
            prob = prob_node.copy()
            print('random walk convergence, iteration: %d' % count)
            break

    # prob作为个收敛矩阵(是按列为概率分布的)，现在按行归一化为概率分布
    print(prob)
    print(prob.sum(axis=0))
    for i in range(prob.shape[0]):
        if prob[i, :].sum() != 0:
            prob[i, :] = prob[i, :] / prob[i, :].sum()
        else:
            if i == 0:
                continue
            prob[i, :] = 1.0 / prob[i, :].shape[0]

    return prob


def main():
    print('now dataset is ', args.dataset)
    Adj_file = 'datasets/' + args.dataset + '/adj_matrix.pkl'

    rw_path = os.path.join(args.rw_path, args.dataset)
    rw_file = 'prob_n' + str(args.anchor_num) + '_restart' + str(args.alpha) + '.pkl'
    rw_file = os.path.join(rw_path, rw_file)
    print('random walk file is: ', rw_file)
    if args.rw == 'w':
        t0 = time.time()
        Trans_adj, Adj_sum, graph = build_graph(Adj_file)
        anchors = anchor_select(graph, anchor_num=args.anchor_num)
        prob_conver = random_walk(Trans_adj, anchors, alpha=args.alpha)
        if os.path.exists(rw_path):
            pickle.dump(prob_conver, open(rw_file, 'wb'))
        else:
            os.mkdir(rw_path)
            pickle.dump(prob_conver, open(rw_file, 'wb'))
        print('random walk spend: %0.4f s' % (time.time() - t0))
    else:
        t0 = time.time()
        print('reading random walk result...')
        prob_conver = pickle.load(open(rw_file, 'rb'))
        print('reading random walk spend: %0.4f s' % (time.time() - t0))


if __name__ == '__main__':
    main()


