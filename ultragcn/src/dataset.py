import os
import pandas as pd
import numpy as np
import torch 
import torch.utils.data as data
import scipy.sparse as sp
from tqdm import tqdm

from src.utils import pload, pstore


def prepare_dataset(args):
    # load label encoded data
    df, n_user, m_item, idx_dict = load_data(args.basepath)

    # train, valid, test split
    train_data, valid_data = random_split(df, args.seed)

    # edge, label, matrix 정의
    interacted_items, test_ground_truth_list, mask, train_mat, constraint_mat = process_data(
        train_data, valid_data, n_user, m_item)

    # Dataloader
    train_loader = data.DataLoader(train_data.values, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = data.DataLoader(list(range(n_user)), batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Compute \Omega to extend UltraGCN to the item-item occurence graph
    if os.path.exists(args.ii_cons_mat_path):
        ii_constraint_mat = pload(args.ii_cons_mat_path)
        ii_neighbor_mat = pload(args.ii_neigh_mat_path)
    else:
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, args.ii_neighbor_num)
        pstore(ii_neighbor_mat, args.ii_neigh_mat_path)
        pstore(ii_constraint_mat, args.ii_cons_mat_path)

 
    return constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, valid_loader, \
            interacted_items, test_ground_truth_list, mask, n_user, m_item


def load_data(basepath):

    # load df
    datapath = os.path.join(basepath, "train_ratings.csv")
    df = pd.read_csv(datapath) 

    df.drop(['time'], axis=1, inplace=True)
    df.drop_duplicates(
        subset=["user", "item"], keep="last", inplace=True
    )

    # user and item label encoding 
    userid, itemid = (
        sorted(list(set(df.user))),
        sorted(list(set(df.item))),
    )
    n_user, m_item = len(userid), len(itemid)

    user2idx = {user: idx for idx, user in enumerate(userid)}
    item2idx = {item: idx for idx, item in enumerate(itemid)}
    idx2user = {idx: user for user, idx in user2idx.items()}
    idx2item = {idx: item for item, idx in item2idx.items()}

    idx_dict = {'user2idx': user2idx, 'item2idx':item2idx, 'idx2user': idx2user, 'idx2item': idx2item}

    df['user'] = df['user'].map(user2idx)
    df['item'] = df['item'].map(item2idx)

    return df, n_user, m_item, idx_dict


def leave_last(df):

    valid = df.groupby('user').tail(1)
    train = df.drop(index=valid.index)
    valid = valid.reset_index(drop=True)

    return train, valid


def random_split(df, seed):

    valid = df.groupby('user').sample(frac=0.1, random_state=seed)
    train = df.drop(index=valid.index)
    valid = valid.reset_index(drop=True)

    return train, valid


def process_data(train_data, valid_data, n_user, m_item):
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32) 

    # user-item interactions saved for each user
    # mask matrix for testing to accelarate testing speed
    interacted_items = [[] for _ in range(n_user)]
    mask = torch.zeros(n_user, m_item)

    print('Collecting train data...')
    total_train=train_data.shape[0]
    with tqdm(total=total_train) as pbar:
        for user, item in zip(train_data.user, train_data.item):
            interacted_items[user].append(item)
            mask[user][item] = -np.inf
            train_mat[user, item] = 1.0
            pbar.update(1)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(n_user)]

    print('Collecting test data...')
    total_valid=valid_data.shape[0]
    with tqdm(total=total_valid) as pbar:
        for user, item in tqdm(zip(valid_data.user, valid_data.item)):
            test_ground_truth_list[user].append(item)
            pbar.update(1)


    # construct degree matrix for graphmf
    items_D = np.sum(train_mat, axis=0).reshape(-1)
    users_D = np.sum(train_mat, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    
    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
    
    return interacted_items, test_ground_truth_list, mask, train_mat, constraint_mat


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)
    # users_D[users_D == 0] == 10 ** -15

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} done'.format(i))
            
    print('Computation \\Omega Done!')
    return res_mat.long(), res_sim_mat.float()