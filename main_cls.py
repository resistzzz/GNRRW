
import os
import argparse
import pickle
import time
import datetime
from model_cls import *
import numpy as np
from data import *
import warnings
warnings.filterwarnings("ignore")


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='yoochoose1_64/diginetica/sample')
'''训练基本参数'''
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_epoch', type=list, default=[3, 6, 9, 12], help='the epoch which the learning rate decay')
# parser.add_argument('--lr_dc_epoch', type=list, default=[5, 10, 15], help='the epoch which the learning rate decay')
parser.add_argument('--patience', type=int, default=5)

'''模型超参数'''
parser.add_argument('--n_factors', type=int, default=40, help='number of anchors')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--routing_iter', type=int, default=4)
parser.add_argument('--hop', type=int, default=3)   # 1 or 2 or 3
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--dropout_gnn', type=float, default=0.2)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--topk', type=list, default=[20], help='topk recommendation')  # [5, 10, 20]

parser.add_argument('--save_path', default='model_save', help='save model root path')
parser.add_argument('--save_epochs', default=[3, 6, 9, 12], type=list)
# parser.add_argument('--save_epochs', default=[5, 10, 15], type=list)

opt = parser.parse_args()
print(opt)


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

if opt.save_path is not None and opt.dataset != 'sample':
    save_path = opt.save_path + '/' + opt.dataset
    save_dir = save_path + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('save dir: ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def main():
    t0 = time.time()
    init_seed(2021)

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj_items = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample) + '.pkl', 'rb'))
    weight_items = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample) + '.pkl', 'rb'))
    param = pickle.load(open('datasets/' + opt.dataset + '/parm' + '.pkl', 'rb'))
    prob = pickle.load(open('rw/' + opt.dataset + '/prob_n' + str(opt.n_factors) + '_restart' + str(opt.alpha) + '.pkl', 'rb'))
    num_items, max_length, item2idx, idx2items = param['num_items'], param['max_length'], param['item2idx'], param[
        'idx2item']

    train_data = transfer2idx(train_data, item2idx)
    test_data = transfer2idx(test_data, item2idx)

    # 按每条session的最长长度进行padding
    train_data = Data(train_data, num_items)
    test_data = Data(test_data, num_items)

    train_slices = train_data.generate_batch(opt.batch_size)
    test_slices = test_data.generate_batch(opt.batch_size)

    adj_items, weight_items = handle_adj(adj_items, weight_items, num_items, opt.sample_num)

    model = NeighRoutingGnnCls2Scores(opt, num_items, adj_items, weight_items, prob, device)

    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_dc_epoch, gamma=opt.lr_dc)

    best_result = {}
    best_epoch = {}
    for k in opt.topk:
        best_result[k] = [0, 0]
        best_epoch[k] = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epochs):
        st = time.time()
        print('-------------------------------------------')
        print('epoch: ', epoch)

        hit, mrr = train_test(model, train_data, test_data, train_slices, test_slices, optimizer, epoch)

        if opt.save_path is not None and epoch in opt.save_epochs and opt.dataset != 'sample':
            save_file = save_dir + '/epoch-' + str(epoch) + '.pt'
            torch.save(model, save_file)
            print('save success! :)')
        flag = 0
        for k in opt.topk:
            if hit[k] > best_result[k][0]:
                best_result[k][0] = hit[k]
                best_epoch[k][0] = epoch
                flag = 1
            if mrr[k] > best_result[k][1]:
                best_result[k][1] = mrr[k]
                best_epoch[k][1] = epoch
                flag = 1
            print('Hit@%d:\t%0.4f %%\tMRR@%d:\t%0.4f %%\t[%0.2f s]' % (k, hit[k], k, mrr[k], (time.time() - st)))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
        scheduler.step()

    print('------------------best result-------------------')
    for k in opt.topk:
        print('Best Result: Hit@%d: %0.4f %%\tMRR@%d: %0.4f %%\t[%0.2f s]' %
              (k, best_result[k][0], k, best_result[k][1], (time.time() - t0)))
        print('Best Epoch: Hit@%d: %d\tMRR@%d: %d\t[%0.2f s]' % (
            k, best_epoch[k][0], k, best_epoch[k][1], (time.time() - t0)))
    print('------------------------------------------------')
    print('Run time: %0.2f s' % (time.time() - t0))


def train_test(model, train_data, test_data, train_slices, test_slices, optimizer, epoch):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = []
    total_rec_loss = []
    total_rec_loss1 = []
    total_rec_loss2 = []

    for index in train_slices:
        optimizer.zero_grad()

        scores, scores1, scores2, targets = forward(model, index, train_data)
        rec_loss1 = model.loss_function1(scores1, targets - 1)
        rec_loss2 = model.loss_function2(scores2, targets - 1)
        rec_loss = model.loss_function(scores, targets - 1)
        loss = rec_loss1 + rec_loss2 + rec_loss
        loss.backward()

        optimizer.step()

        total_loss.append(loss.item())
        total_rec_loss1.append(rec_loss1.item())
        total_rec_loss2.append(rec_loss2.item())
        total_rec_loss.append(rec_loss.item())

    print('Loss:\t%.3f\tlr:\t%0.6f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Rec1:\t%.3f\tlr:\t%0.6f' % (np.mean(total_rec_loss1), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Rec2:\t%.3f\tlr:\t%0.6f' % (np.mean(total_rec_loss2), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Rec:\t%.3f\tlr:\t%0.6f' % (np.mean(total_rec_loss), optimizer.state_dict()['param_groups'][0]['lr']))

    print('----------------')
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit_dic, mrr_dic = {}, {}
    hit_seq, mrr_seq = {}, {}
    hit_cls, mrr_cls = {}, {}
    for k in opt.topk:
        hit_dic[k] = []
        mrr_dic[k] = []
        hit_seq[k] = []
        mrr_seq[k] = []
        hit_cls[k] = []
        mrr_cls[k] = []
    with torch.no_grad():
        for index in test_slices:

            tes_scores, tes_scores1, tes_scores2, tes_targets = forward(model, index, test_data)

            for k in opt.topk:
                predict = tes_scores.cpu().topk(k)[1]
                predict = predict.cpu()
                for pred, target in zip(predict, tes_targets.cpu()):
                    hit_dic[k].append(np.isin(target - 1, pred))
                    if len(np.where(pred == target - 1)[0]) == 0:
                        mrr_dic[k].append(0)
                    else:
                        mrr_dic[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
            for k in opt.topk:
                predict = tes_scores1.cpu().topk(k)[1]
                predict = predict.cpu()
                for pred, target in zip(predict, tes_targets.cpu()):
                    hit_seq[k].append(np.isin(target - 1, pred))
                    if len(np.where(pred == target - 1)[0]) == 0:
                        mrr_seq[k].append(0)
                    else:
                        mrr_seq[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
            for k in opt.topk:
                predict = tes_scores2.cpu().topk(k)[1]
                predict = predict.cpu()
                for pred, target in zip(predict, tes_targets.cpu()):
                    hit_cls[k].append(np.isin(target - 1, pred))
                    if len(np.where(pred == target - 1)[0]) == 0:
                        mrr_cls[k].append(0)
                    else:
                        mrr_cls[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))

        for k in opt.topk:
            hit_dic[k] = np.mean(hit_dic[k]) * 100
            mrr_dic[k] = np.mean(mrr_dic[k]) * 100
            hit_seq[k] = np.mean(hit_seq[k]) * 100
            mrr_seq[k] = np.mean(mrr_seq[k]) * 100
            hit_cls[k] = np.mean(hit_cls[k]) * 100
            mrr_cls[k] = np.mean(mrr_cls[k]) * 100
            print('HitSeq@%d:\t%0.4f %%\tMRRSeq@%d:\t%0.4f %%\t' % (k, hit_seq[k], k, mrr_seq[k]))
            print('HitCls@%d:\t%0.4f %%\tMRRCls@%d:\t%0.4f %%\t' % (k, hit_cls[k], k, mrr_cls[k]))

        return hit_dic, mrr_dic


def forward(model, index, data):
    inp_sess, targets, mask_1, mask_inf, lengths = data.get_slice_sess_mask(index)
    inp_sess = torch.LongTensor(inp_sess).to(device)
    lengths = torch.LongTensor(lengths).to(device)
    targets = torch.LongTensor(targets).to(device)
    mask_1 = torch.FloatTensor(mask_1).to(device)
    mask_inf = torch.FloatTensor(mask_inf).to(device)

    scores, scores1, scores2 = model(inp_sess, mask_1, mask_inf, lengths)

    return scores, scores1, scores2, targets


if __name__ == '__main__':
    main()


