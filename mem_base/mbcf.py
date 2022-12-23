import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch

import argparse
from tqdm import tqdm
tqdm.pandas()

import os
import random
import copy

from utils import (
    generate_submission_file,
    set_seed,
    recall_at_k,
    ndcg_k
)


def get_full_sort_score(answers, pred_list): # baseline trainer에 있는 것 그대로 가져옴
    recall, ndcg = [], []
    for k in [5, 10]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
    post_fix = {
        "RECALL@5": "{:.4f}".format(recall[0]),
        "NDCG@5": "{:.4f}".format(ndcg[0]),
        "RECALL@10": "{:.4f}".format(recall[1]),
        "NDCG@10": "{:.4f}".format(ndcg[1]),
    }
    print(post_fix)

    return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)


def main(args):

    print('Load data... ', end='', flush=True)
    ratings = pd.read_csv('../data/train/train_ratings.csv')
    user2idx = {v: k for k, v in enumerate(ratings['user'].unique())}
    idx2user = {k: v for k, v in enumerate(ratings['user'].unique())}

    item2idx = {v: k for k, v in enumerate(ratings['item'].unique())}
    idx2item = {k: v for k, v in enumerate(ratings['item'].unique())}

    ratings['user'] = ratings['user'].map(user2idx)
    ratings['item'] = ratings['item'].map(item2idx)

    user_item_seq = ratings.groupby('user')['item'].apply(list)
    print('done.')

    # 라벨 인코딩을 했으므로 31,000 by 6,800 -> 메모리 많이 아낌
    rating_matrix = torch.zeros((ratings['user'].nunique(), ratings['item'].nunique()), dtype=torch.float16) # float16 해야 메모리 절약
    if not args.inference:
        print('Make validation matrix... ', flush=True)
        for u, items in tqdm(user_item_seq.iteritems(), total=user_item_seq.shape[0]):
            for i in items[:-2]: # 유저별 맨뒤 두개 빵꾸
                rating_matrix[u][i] = 1.
    else:
        print('Make submission matrix... ', flush=True)
        for u, items in tqdm(user_item_seq.iteritems(), total=user_item_seq.shape[0]):
            for i in items:
                rating_matrix[u][i] = 1.
    print('done.')

    rating_matrix_cuda = rating_matrix.to('cuda')
    
    # 유사도는 각 벡터의 내적값을 함. 0, 1 로만 이루어져 있기 때문에 가능함
    # 원래 내적 자체도 유사도역할을 할 수 있음. 코사인 유사도는 각도에 더 중점
    # 서로 수직인 벡터는 내적값이 0, 반대면 음수임.
    # 같은 방향이면 내적값이 가장 큼. -> 코사인 유사도는 이를 정규화해 길이가 달라도 같도록 함. 내적은 길이 차이에 따른 값차이 당연히 있음.
    # 그래도 0, 1 로만 이루어져 있어서 따로 그런 정규화 안해도 됨.
    if args.item_base:
        # 아이템 베이스드로 하면 열을 기준으로 유사도 구함.
        item_sim_matrix_cuda = torch.matmul(rating_matrix_cuda.T, rating_matrix_cuda)
        item_sim_sorted_matrix_cuda = torch.argsort(item_sim_matrix_cuda, dim=-1, descending=True)

        del item_sim_matrix_cuda
        torch.cuda.empty_cache() 

        neighbors_cuda = item_sim_sorted_matrix_cuda[:, 1:args.k+1]
        # rating_matrix_cuda = rating_matrix_cuda.T

        print(f'Get top {args.k} item base... ', flush=True)

        new_ratings = torch.zeros((ratings['user'].nunique(), ratings['item'].nunique()), dtype=torch.float16).to('cuda')
        for i in tqdm(range(rating_matrix_cuda.shape[1])):
            new_rating = torch.mean(rating_matrix_cuda[:, neighbors_cuda[i]], dim=1)
            already_view = torch.where(rating_matrix_cuda[:, i] > 0)[0] 
            # 이미 본 것들 0으로 만드는데, 여기서 말고 나중에 다 만들고 본거 빼는게 나을것 같기도?
            new_rating[already_view] = 0.
            new_ratings[:, i] = new_rating
        
        preds = torch.argsort(new_ratings, descending=True, dim=1).detach().cpu().numpy()
        print('done.')
    
    else:
        # 유저 베이스 행 기준
        user_sim_matrix_cuda = torch.matmul(rating_matrix_cuda, rating_matrix_cuda.T)
        user_sim_sorted_matrix_cuda = torch.argsort(user_sim_matrix_cuda, dim=-1, descending=True)

        del user_sim_matrix_cuda
        torch.cuda.empty_cache() 

        neighbors_cuda = user_sim_sorted_matrix_cuda[:, 1:args.k+1]

        print(f'Get top {args.k} user base... ', flush=True)

        new_ratings = torch.zeros((ratings['user'].nunique(), ratings['item'].nunique()), dtype=torch.float16).to('cuda')
        for u in tqdm(range(rating_matrix_cuda.shape[0])):
            new_rating = torch.mean(rating_matrix_cuda[neighbors_cuda[u]], dim=0)
            already_view = torch.where(rating_matrix_cuda[u] > 0)[0]

            new_rating[already_view] = 0.
            new_ratings[u, :] = new_rating
        
        preds = torch.argsort(new_ratings, descending=True, dim=1).detach().cpu().numpy()
        print('done.')

    # validation 마지막 두개
    if not args.inference:
        # valid 는 라벨인코딩 된 상태에서 진행해도 무방함.
        answers = [item_list[-2:] for item_list in user_item_seq]
        get_full_sort_score(answers, preds)
    else:
        # submission 은 라벨인코딩 된걸 되돌려야 해서 밑의 과정이 필요함.
        if not os.path.exists('output/'):
            os.mkdir('output/')
        item_preds = []
        print('Submission label encoding... ', flush=True)
        for pred in tqdm(preds):
            item_pred = []
            for idx in pred:
                item_pred.append(idx2item[idx]) # 되돌리기
            item_preds.append(item_pred)
        item_preds = np.array(item_preds)
        generate_submission_file(args.data_path, item_preds[:, :10])
        print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--item_base", default=0, type=int) # dkt 의 leak 처럼 1 써주면 작동. 
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--inference", default=0, type=int) # dkt 의 leak 처럼 1 써주면 작동. 
    
    args = parser.parse_args()
    args.data_path = args.data_dir + "train_ratings.csv"

    set_seed(42)
    main(args)