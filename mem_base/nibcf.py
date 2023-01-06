import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch

import argparse
from tqdm import tqdm
tqdm.pandas()

import os
import random
import datetime

from utils import (
    set_seed,
    get_full_sort_score,
    get_sub_dataset,
    generate_submission_file,
)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--valid', default=1, type=int)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--inference", default=False, action='store_true') 

    args = parser.parse_args()
    args.data_path = args.data_dir + "train_ratings.csv"

    args.name = 'ibcf'
    return args


def main(args):
    set_seed(args.seed)

    ratings = pd.read_csv(args.data_path)
    user2idx = {v: k for k, v in enumerate(ratings['user'].unique())}
    idx2user = {k: v for k, v in enumerate(ratings['user'].unique())}

    item2idx = {v: k for k, v in enumerate(ratings['item'].unique())}
    idx2item = {k: v for k, v in enumerate(ratings['item'].unique())}

    ratings['user'] = ratings['user'].map(user2idx)
    ratings['item'] = ratings['item'].map(item2idx)

    args.n_users = ratings['user'].nunique()
    args.n_items = ratings['item'].nunique()

    if not args.inference: # valid 
        ratings, answers = get_sub_dataset(ratings, args.valid)
    
    positives = ratings.groupby('user')['item'].apply(list)
    
    if not os.path.exists('asset/'):
        os.mkdir('asset/')
    
    if not args.inference:
        if not os.path.exists(f'asset/rating_mat_v{args.valid}_seed{args.seed}.npy'):
            rating_mat = torch.zeros((args.n_users, args.n_items), dtype=torch.float16)
            for u, items in tqdm(positives.iteritems(), total=len(positives), desc=f'make valid rating mat'):
                for i in items:
                    rating_mat[u][i] = 1.0
            np.save(f'asset/rating_mat_v{args.valid}_seed{args.seed}.npy', rating_mat.numpy())
        else:
            rating_mat = np.load(f'asset/rating_mat_v{args.valid}_seed{args.seed}.npy')
            rating_mat = torch.from_numpy(rating_mat).half()
    else:
        rating_mat = torch.zeros((args.n_users, args.n_items), dtype=torch.float16) # 32 면 느려지나
        for u, items in tqdm(positives.iteritems(), total=len(positives), desc='make inference rating mat'):
            for i in items:
                rating_mat[u][i] = 1.0

    rating_mat = rating_mat.to(args.device)
    base_rating_mat = rating_mat.clone()
    # aka 매그니튜드, 많이 본 유저일 수록 값을 낮춘다.
    # 아이템 베이스드에서 사용함.
    rating_mat = rating_mat / rating_mat.norm(dim=1, keepdim=True)
    
    item_sim_mat = torch.matmul(rating_mat.T, rating_mat)
    item_sim_norm_vec = torch.norm(rating_mat, dim=0, keepdim=True)
    item_sim_norm_mat = torch.matmul(item_sim_norm_vec.T, item_sim_norm_vec)
    item_sim_mat = item_sim_mat / (item_sim_norm_mat + 1e-8)

    del item_sim_norm_mat
    torch.cuda.empty_cache() 

    sims, neighbors = torch.sort(item_sim_mat, dim=1, descending=True)
    sims = sims[:, 1:args.k + 1]
    neighbors = neighbors[:,  1:args.k + 1]
    del item_sim_mat
    torch.cuda.empty_cache()

    expand_neighbors = neighbors.unsqueeze(1).expand(args.n_items, args.n_users, args.k)
    
    pred_rating_mat = torch.zeros_like(rating_mat).to(args.device)
    for i in tqdm(range(args.n_items), desc='make pred rating mat'):
        rs = torch.gather(base_rating_mat, 1, expand_neighbors[i])
        nomi = torch.sum(rs * sims[i], dim=1)
        denomi = torch.sum(sims[i])
        pred_rating_mat[:, i] = nomi / denomi

    pred_rating_mat[rating_mat > 0] = 0.0
    preds = torch.argsort(pred_rating_mat, descending=True, dim=1).cpu().numpy()

    if not args.inference:
        print(f'item_based_cf, k = {args.k}, valid = {args.valid}', flush=True)

    if not args.inference:
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
        time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')
        generate_submission_file(args.data_path, item_preds[:, :10], args.name, time_info, args.k)
        print('done.')


if __name__ == '__main__':
    main(get_parser())