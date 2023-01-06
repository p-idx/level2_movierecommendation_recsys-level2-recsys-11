import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from typing import Tuple
from tqdm import tqdm
tqdm.pandas()

import os
import random
import copy
import pickle

from utils import (
    set_seed,
    get_full_sort_score,
    get_sub_dataset,
    generate_submission_file,
    NegativeSampler
)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--valid', default=1, type=int)

    
    parser.add_argument("--sampler", default='uni', choices=['uni', 'freq'], type=str)
    parser.add_argument("--freq_rate", default=0.5, type=float)

    parser.add_argument("--neg", default=5, type=int)
    parser.add_argument("--f", default=64, type=int)
    parser.add_argument("--alpha", default=1, type=int)
    parser.add_argument("--r", default=0.1, type=int)
    parser.add_argument("--mean", default=False, action='store_true') 

    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("--batch_size", default=4096, type=int)

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--inference", default=False, action='store_true') 

    args = parser.parse_args()
    args.data_path = args.data_dir + "train_ratings.csv"

    args.name = 'mf-bpr'
    return args


class BPRDataset(Dataset):
    def __init__(
        self,
        ratings: pd.DataFrame,
        args
    ):
        self.triplets = []
        self.positives = ratings.groupby('user')['item'].apply(list)

        self.negative_sampler = NegativeSampler(ratings, self.positives, args.freq_rate)
        self.negative_sampling(args)

    
    def negative_sampling(self, args):
        self.triplets = []
        if args.sampler == 'uni':
            sampler_name='uni'
            sampling = self.negative_sampler.uniform_sampling
            cache_file_name = f'asset/triplets_{sampler_name}_n{args.neg}_seed{args.seed}.pickle'

        elif args.sampler == 'freq':
            sampler_name='freq'
            sampling = self.negative_sampler.freq_sampling
            cache_file_name = f'asset/triplets_{sampler_name}_freq{args.freq_rate:.2f}_n{args.neg}_seed{args.seed}.pickle'

        else:
            raise 'sampler error'

        for u, pos_items in tqdm(self.positives.iteritems(), total=len(self.positives), desc=f'{args.sampler} {args.neg} negative sampling'):
            for pos in pos_items:
                neg_items = sampling(u, args.neg)
                for neg in neg_items:
                    self.triplets.append((u, pos, neg))
                    
        self.triplets = torch.LongTensor(self.triplets)


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]


class BPR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_dim = args.f

        self.lu = args.r
        self.lp = args.r
        self.ln = args.r

        self.n_user = args.n_users
        self.n_item = args.n_items
        self.user_embedding = nn.Embedding(args.n_users, self.feature_dim, dtype=torch.float64)
        self.item_embedding = nn.Embedding(args.n_items, self.feature_dim, dtype=torch.float64)
        self.args = args

    def forward(self, triplets: torch.tensor) -> torch.tensor:
        users = self.user_embedding(triplets[:, 0])
        pos_items = self.item_embedding(triplets[:, 1])
        neg_items = self.item_embedding(triplets[:, 2])

        r_ui = torch.sum((users * pos_items), dim=-1)
        r_uj = torch.sum((users * neg_items), dim=-1)

        r_uij = r_ui - r_uj
        losses = torch.log(torch.sigmoid(r_uij))
        
        regs = self.args.r * \
            (torch.pow(torch.norm(users, dim=1), 2) \
            + torch.pow(torch.norm(pos_items, dim=1), 2) \
            + torch.pow(torch.norm(neg_items, dim=1), 2))

        loss = losses - regs

        if self.args.mean:
            return - torch.mean(loss)
        else:
            return - torch.sum(loss)


def main(args):
    set_seed(args.seed)
    ratings = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv').sort_values(by=['user', 'time'])
    
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

    train_dataset = BPRDataset(ratings, args)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset, replacement=True), # bootstrap
        batch_size=args.batch_size,
        num_workers=8
    )

    print(f'{args.name}, f: {args.f}, neg: {args.neg}, batch_size: {args.batch_size}', flush=True)

    model = BPR(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss = []
    for epoch in range(args.iter):
        train_dataset.negative_sampling(args)
        model.train()
        batch_loss = []
        for i, triplets in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'epoch {epoch}'):
            
            triplets = triplets.to('cuda')
            loss = model(triplets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        
        print(f'epoch: {epoch}', f'loss: {np.mean(batch_loss)}', flush=True)
        
        model.eval()
        with torch.no_grad():
            X = model.user_embedding.weight.data
            Y = model.item_embedding.weight.data
        
            pred_rating_mat = torch.matmul(X, Y.T)
            pred_rating_mat[rating_mat > 0] = -999.0
            _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)
            preds = recs.cpu().numpy()
            get_full_sort_score(answers, preds)

        
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