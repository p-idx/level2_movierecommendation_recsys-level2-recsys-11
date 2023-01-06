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

    parser.add_argument("--neg", default=4, type=int)
    parser.add_argument("--sampler", default='uni', choices=['uni', 'freq'], type=str)

    parser.add_argument('--model', choices=['gmf', 'mlp', 'nmf'], type=str)
    parser.add_argument('--pretrain', default=False, action='store_true')

    parser.add_argument("--gmf_f", default=8, type=int)
    parser.add_argument("--mlp_f", default=32, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)

    parser.add_argument('--layers', nargs='+', default=[32, 16, 8])
    # parser.add_argument("--alpha", default=1, type=int)
    # parser.add_argument("--r", default=0.1, type=int)
    parser.add_argument("--mean", default=False, action='store_true') 

    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--iter", default=50, type=int)
    parser.add_argument("--batch_size", default=4096, type=int)

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--inference", default=False, action='store_true') 

    args = parser.parse_args()
    args.data_path = args.data_dir + "train_ratings.csv"

    args.name = args.model
    return args


class NMFDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame, args):
        self.ratings = ratings
        self.positives = ratings.groupby('user')['item'].apply(list)
        self.args = args
        self.datas = []

        self.n_items = ratings['item'].nunique()
        self.negative_sampler = NegativeSampler(ratings, self.positives, 0.2)
        # self.negative_sampling(args)
    
    def negative_sampling(self, args):
        self.datas = []
        if args.sampler == 'uni':
            sampler_name='uni'
            sampling = self.negative_sampler.uniform_sampling
            cache_file_name = f'asset/triplets_{sampler_name}_n{args.neg}_seed{args.seed}.pickle'

        elif args.sampler == 'freq':
            sampler_name='freq'
            sampling = self.negative_sampler.freq_sampling
            cache_file_name = f'asset/triplets_{sampler_name}_freq{args.freq_rate:.2f}_n{args.neg}_seed{args.seed}.pickle'

        for u, pos_items in tqdm(self.positives.iteritems(), total=len(self.positives), desc=f'pos-neg({args.neg})sampling'):
            for pos_item in pos_items:
                self.datas.append([u, pos_item, 1])
                neg_items = sampling(u, args.neg)
                for neg_item in neg_items:
                    self.datas.append([u, neg_item, 0])

        self.datas = torch.LongTensor(self.datas)
        
    def __len__(self):
        return len(self.ratings) * (self.args.neg + 1)

    def __getitem__(self, index):
        return self.datas[index]


class NueMF(nn.Module):
    def __init__(self, args, gmf_model=None, mlp_model=None):
        super().__init__()
        self.args = args
        

        self.n_users = args.n_users
        self.n_items = args.n_items

        # GMF
        self.GMF_user_embedding = nn.Embedding(self.n_users, args.gmf_f)
        self.GMF_item_embedding = nn.Embedding(self.n_items, args.gmf_f)
        self.GMF_fc = nn.Linear(args.gmf_f, 1)

        # MLP
        self.MLP_user_embedding = nn.Embedding(self.n_users, args.mlp_f)
        self.MLP_item_embedding = nn.Embedding(self.n_items, args.mlp_f)

        mlp_layers = [nn.Linear(args.mlp_f * 2, args.layers[0]), nn.ReLU()]
        for i in range(len(args.layers) - 1):
            mlp_layers.append(nn.Dropout(args.dropout))
            mlp_layers.append(nn.Linear(args.layers[i], args.layers[i+1]))
            mlp_layers.append(nn.ReLU())
        
        self.MLP_layers = nn.Sequential(*mlp_layers)
        self.MLP_fc = nn.Linear(args.layers[-1], 1)
        
        # NeuMF
        self.NMF_fc = nn.Linear(args.gmf_f + args.layers[-1], 1)

        if args.model == 'nmf' and args.pretrain:
            # GMF 갈아끼우기
            self.gmf_model = gmf_model
            self.GMF_user_embedding.weight.data.copy_(self.gmf_model.GMF_user_embedding.weight)
            self.GMF_item_embedding.weight.data.copy_(self.gmf_model.GMF_item_embedding.weight)

            self.GMF_fc.weight.data.copy_(self.gmf_model.GMF_fc.weight)
            self.GMF_fc.bias.data.copy_(self.gmf_model.GMF_fc.bias)

            # MLP 갈아끼우기
            self.mlp_model = mlp_model
            self.MLP_user_embedding.weight.data.copy_(self.mlp_model.MLP_user_embedding.weight)
            self.MLP_item_embedding.weight.data.copy_(self.mlp_model.MLP_item_embedding.weight)

            for m1, m2 in zip(self.MLP_layers, self.mlp_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            
            self.MLP_fc.weight.data.copy_(self.mlp_model.MLP_fc.weight)
            self.MLP_fc.bias.data.copy_(self.mlp_model.MLP_fc.bias)

    		# predict layers
            nmf_fc_weight = torch.cat([
				self.gmf_model.GMF_fc.weight, 
				self.mlp_model.MLP_fc.weight], dim=1)
            nmf_fc_bias = self.gmf_model.GMF_fc.bias + \
						self.mlp_model.MLP_fc.bias

            self.NMF_fc.weight.data.copy_(0.5 * nmf_fc_weight)
            self.NMF_fc.bias.data.copy_(0.5 * nmf_fc_bias)


    def forward(self, user, item):

        if self.args.model != 'mlp':
            gmf_user_emb = self.GMF_user_embedding(user)
            gmf_item_emb = self.GMF_item_embedding(item)
            gmf_out = torch.mul(gmf_user_emb, gmf_item_emb) # (B, gmf_f)

        if self.args.model != 'gmf':
            mlp_user_emb = self.MLP_user_embedding(user)
            mlp_item_emb = self.MLP_item_embedding(item)
            mlp_out = self.MLP_layers(torch.cat([mlp_user_emb, mlp_item_emb], dim=1)) # (B, mlp_f)

        
        if self.args.model == 'gmf':
            return self.GMF_fc(gmf_out).squeeze()
        if self.args.model == 'mlp':
            return self.MLP_fc(mlp_out).squeeze()
        
        return self.NMF_fc(torch.cat([gmf_out, mlp_out], dim=1)).squeeze()


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

    train_dataset = NMFDataset(ratings, args)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=8
    )

    
    print(f'{args.name}, gmf_f: {args.gmf_f}, mlp_f: {args.mlp_f}, neg: {args.neg}, batch_size: {args.batch_size}', flush=True)

    if not args.pretrain:
        model = NueMF(args).to(args.device)
    else:
        gmf_model = torch.load('model/gmf_train.pt').to(args.device)
        mlp_model = torch.load('model/mlp_train.pt').to(args.device)
        
        model = NueMF(args, gmf_model, mlp_model).to(args.device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) \
        # if not args.pretrain else torch.optim.SGD(model.parameters(), lr=args.lr)

    if not os.path.exists('model/'):
        os.mkdir('model/')

    train_recall = 0.0
    loss = []
    for epoch in range(args.iter):
        train_dataset.negative_sampling(args)
        model.train()
        batch_loss = []
        for i, datas in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'epoch {epoch}'):
            users = datas[:, 0].to(args.device)
            items = datas[:, 1].to(args.device)
            model_answers = datas[:, 2].float().to(args.device)

            preds = model(users, items)
            # print(preds, model_answers)
            loss = criterion(preds, model_answers)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        
        print(f'epoch: {epoch}', f'loss: {np.mean(batch_loss)}', flush=True)
        
        if not args.inference:
            model.eval()
            with torch.no_grad():
                pred_rating_mat = torch.empty_like(rating_mat)
                for u in tqdm(range(args.n_users), desc='eval'):
                    user = torch.ones(args.n_items, dtype=torch.long).to(args.device) * u
                    items = torch.arange(args.n_items, dtype=torch.long).to(args.device)
                    preds = model(user, items)
                    preds = torch.sigmoid(preds)
                    pred_rating_mat[u] = preds

                pred_rating_mat[rating_mat > 0] = -999.0
                _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)
                preds = recs.cpu().numpy()
                metrics, _= get_full_sort_score(answers, preds)
                if metrics[2] > train_recall:
                    train_recall = metrics[2]
                    if args.model == 'gmf':
                        torch.save(model, 'model/gmf_train.pt')
                    elif args.model == 'mlp':
                        torch.save(model, 'model/mlp_train.pt')
        else:
            if args.model == 'gmf':
                torch.save(model, 'model/gmf_inference.pt')

            elif args.model == 'mlp':
                torch.save(model, 'model/mlp_inference.pt')    
            




if __name__ == '__main__':
    main(get_parser())
