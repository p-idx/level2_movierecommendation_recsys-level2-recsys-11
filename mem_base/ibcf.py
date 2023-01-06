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
    ratings = pd.read_csv(args.data_path)
    user2idx = {v: k for k, v in enumerate(ratings['user'].unique())}
    idx2user = {k: v for k, v in enumerate(ratings['user'].unique())}

    item2idx = {v: k for k, v in enumerate(ratings['item'].unique())}
    idx2item = {k: v for k, v in enumerate(ratings['item'].unique())}

    ratings['user'] = ratings['user'].map(user2idx)
    ratings['item'] = ratings['item'].map(item2idx)

    user_item_seq = ratings.groupby('user')['item'].apply(list)
    print('done.')

    # 라벨 인코딩을 했으므로 31,000 by 6,800 -> 메모리 많이 아낌
    try:
        if not args.inference:
            print('Get validation matrix... ', flush=True)
            answers = np.load('validation_answers_ibcf.npy').tolist()
            rating_matrix = torch.from_numpy(np.load('validation_matrix_ibcf.npy')).float()
        else:
            print('Get submission matrix... ', flush=True)
            rating_matrix = torch.from_numpy(np.load('submission_matrix_ibcf.npy')).float()
    except:
        print('Make matrix...', flush=True)
        rating_matrix = torch.zeros((ratings['user'].nunique(), ratings['item'].nunique()), dtype=torch.float32) # float16 해야 메모리 절약
        if not args.inference:
            answers = []
            for u, items in tqdm(user_item_seq.iteritems(), total=user_item_seq.shape[0]):
                last_one = items[-1] # 맨마지막 한개
                shuffles = items[:-1]
                random.shuffle(shuffles)
                # 셔플 후 뒤에서 두개
                answer = shuffles[-3:]
                answer.append(last_one)
                answers.append(answer)
                for i in shuffles[:-3]: # 유저별 맨뒤 두개 빵꾸
                    rating_matrix[u][i] = 1.

            np.save('validation_matrix_ibcf.npy', rating_matrix.cpu().numpy())
            np.save('validation_answers_ibcf.npy', np.array(answers, dtype=int))
        else:
            for u, items in tqdm(user_item_seq.iteritems(), total=user_item_seq.shape[0]):
                for i in items:
                    rating_matrix[u][i] = 1.
            np.save('submission_matrix_ibcf.npy', rating_matrix.cpu().numpy())

    print('done.')

    rating_matrix = rating_matrix / torch.norm(rating_matrix, dim=1, keepdim=True)
    # 매그니튜드 적용
    item_sim_matrix = torch.matmul(rating_matrix.T, rating_matrix)
    item_sim_norm = torch.norm(rating_matrix.T, p=2, dim=1, keepdim=True)
    item_norm_matrix = torch.matmul(item_sim_norm, item_sim_norm.T)

    item_sim_matrix = item_sim_matrix / (item_norm_matrix + 1e-8)

    neighbor_item_matrix = \
        item_sim_matrix.argsort(dim=-1, descending=True)

    neighbor_item_matrix = neighbor_item_matrix[:, :args.k]

    preds = []
    for user in tqdm(range(rating_matrix.shape[0])):    
        known_user_likes = torch.where(rating_matrix[user] > 0)[0]
        most_sim_to_likes = \
            neighbor_item_matrix[known_user_likes]
        sim_list = most_sim_to_likes.numpy()

        sim_list = \
            sorted(list(set([item for sublist in sim_list for item in sublist])))

        idx2sim = {k: v for k, v in enumerate(sim_list)} # 영화 번호 다시 생성용

        # sim_list 끼리의 이웃 유사도로만 계산하여 추천
        neighbors = item_sim_matrix[sim_list][:, sim_list] # (sim_list, sim_list)

        
        user_vector = rating_matrix[user, sim_list] # (sim_list, )
        score = neighbors.matmul(user_vector) /\
            torch.sum(neighbors, dim=-1) # score 구하는 공식이 이럼.

        indices = torch.argsort(score, descending=True)
        recs = list(map(lambda idx: idx2sim[idx], indices.tolist()))
        recs = [rec for rec in recs if rec not in known_user_likes]
        preds.append(recs)


    # validation 마지막 두개
    if not args.inference:
        # valid 는 라벨인코딩 된 상태에서 진행해도 무방함.
        get_full_sort_score(answers, preds)
    else:
        # submission 은 라벨인코딩 된걸 되돌려야 해서 밑의 과정이 필요함.
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
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

    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--inference", default=0, type=int) # dkt 의 leak 처럼 1 써주면 작동. 
    
    args = parser.parse_args()
    args.data_path = args.data_dir + "train_ratings.csv"

    set_seed(42)
    main(args)