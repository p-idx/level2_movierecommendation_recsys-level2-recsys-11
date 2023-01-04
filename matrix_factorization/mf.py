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

    
    parser.add_argument("--f", default=8, type=int)
    parser.add_argument("--alpha", default=1, type=int)
    parser.add_argument("--l1", default=0.5, type=int)
    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--inference", default=False, action='store_true') 

    args = parser.parse_args()
    args.data_path = args.data_dir + "train_ratings.csv"

    args.name = 'mf-als'
    return args


def loss(rating_mat, X, Y, C, l1):
    loss_front = torch.sum(torch.mul(C, torch.pow(rating_mat - torch.matmul(X, Y.T), 2)))
    loss_back = l1 * (torch.sum(torch.mul(X, X)) + torch.sum(torch.mul(Y, Y)))
    return loss_front + loss_back


def als(rating_mat: torch.tensor, answers: list, alpha=1, iter=10, l1=0.1, feature_dim=8, device='cuda', inference=False):
    # x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
    # y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

        with torch.no_grad():
            C = rating_mat * alpha + 1
            P = rating_mat

            user_size = rating_mat.size(0)
            item_size = rating_mat.size(1)

            X = torch.rand((user_size, feature_dim), dtype=torch.float32).to(device)
            Y = torch.rand((item_size, feature_dim), dtype=torch.float32).to(device)

            # X_I = torch.eye(user_size)
            # Y_I = torch.eye(item_size)

            I = torch.eye(feature_dim, dtype=torch.float32).to(device)
            lI = l1 * I

            pred_rating_mat = torch.matmul(X, Y.T)
            pred_rating_mat[rating_mat > 0] = 0.0
            _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)
            preds = recs.cpu().numpy()

            
            print('random init.')
            print(f'loss: {loss(rating_mat, X, Y, C, l1)}')
            if not inference:
                get_full_sort_score(answers, preds)

            for it in range(iter):
                # xTx = X.T.matmul(X)
                # yTy = Y.T.matmul(Y)

                for u in tqdm(range(user_size)):
                    # Cu = torch.diag(C[u])
                    left = Y.T.mul(C[u]).matmul(Y) + lI
                    right = Y.T.mul(C[u]).matmul(P[u])
                    X[u] = torch.linalg.solve(left, right)

                for i in tqdm(range(item_size)):
                    # Ci = torch.diag(C[:, i])
                    left = X.T.mul(C[:, i]).matmul(X) + lI
                    right = X.T.mul(C[:, i]).matmul(P[:, i])
                    Y[i] = torch.linalg.solve(left, right)

                pred_rating_mat = torch.matmul(X, Y.T)
                pred_rating_mat[rating_mat > 0] = 0.0
                _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)
                preds = recs.cpu().numpy()
                
                print('iter', it+1)
                print(f'loss: {loss(rating_mat, X, Y, C, l1)}')
                if not inference:
                    get_full_sort_score(answers, preds)

        return preds



def main(args):
    set_seed(args.seed)

    ratings = pd.read_csv('../../data/train/train_ratings.csv')
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
            rating_mat = torch.zeros((args.n_users, args.n_items), dtype=torch.float32)
            for u, items in tqdm(positives.iteritems(), total=len(positives), desc=f'make valid rating mat'):
                for i in items:
                    rating_mat[u][i] = 1.0
            np.save(f'asset/rating_mat_v{args.valid}_seed{args.seed}.npy', rating_mat.numpy())
        else:
            rating_mat = np.load(f'asset/rating_mat_v{args.valid}_seed{args.seed}.npy')
            rating_mat = torch.from_numpy(rating_mat)
    else:
        rating_mat = torch.zeros((args.n_users, args.n_items), dtype=torch.float32) # 32 면 느려지나
        for u, items in tqdm(positives.iteritems(), total=len(positives), desc='make inference rating mat'):
            for i in items:
                rating_mat[u][i] = 1.0

    rating_mat = rating_mat.to(args.device)
    print('feature_dim:', args.f, 'alpha:', args.alpha, 'l1:', args.l1)

    preds = als(
        rating_mat, 
        answers, 
        alpha=args.alpha, 
        l1= args.l1, 
        feature_dim=args.f, 
        inference=args.inference
    )
    
    if args.inference:
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
    main(get_parser())