import os
import pandas as pd
import numpy as np
import torch 
import torch.utils.data as data
import scipy.sparse as sp



def prepare_dataset(device, params):
    train_data, test_data = load_data(params['basepath'], params['fe_num'])

    # 전체 유저와 문제 인덱싱
    user2idx, item2idx, n_user, m_item = indexing_data(train_data, test_data)
    params['user_num'] = n_user
    params['item_num'] = m_item

    # train, valid, test split
    train_data, valid_data, test_data = separate_data(train_data, test_data)

    # edge, label, matrix 정의
    pos_edges, neg_edges, train_edges, valid_edges, valid_label, test_edges, train_mat, constraint_mat = process_data(train_data, valid_data, test_data, user2idx, item2idx, n_user, m_item, device)

    # Dataloader
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    train_loader = data.DataLoader(train_edges, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_edges, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_edges, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # mask matrix for testing to accelerate testing speed
    # when doing topk 
    # mask = torch.zeros(n_user, m_item)
    # interacted_items = [[] for _ in range(n_user)]
    # for (u,i) in train_data:
    #     mask[u][i] = -np.inf
    #     interacted_items[u].append(i)


    # test user-item interaction, which is ground truth
    # test_ground_truth_list = [[] for _ in range(n_user)]
    # for (u, i) in test_data:
    #     test_ground_truth_list[u].append(i)

    # Compute \Omega to extend UltraGCN to the item-item occurence graph
    ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, params['ii_neighbor_num'])
 
    return constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, valid_loader, pos_edges, neg_edges, valid_label, params


def load_data(basepath, fe_num):
    path1 = os.path.join(basepath, f"FE{fe_num}", "train_data.csv")
    path2 = os.path.join(basepath, f"FE{fe_num}", "test_data.csv")
    train = pd.read_csv(path1) # merged_train
    test = pd.read_csv(path2)

    train.drop_duplicates(
        subset=["userID", "assessmentItemID_c"], keep="last", inplace=True
    )

    return train, test


def indexing_data(train, test):
    data = pd.concat([train,test])

    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID_c))),
    )
    n_user, m_item = len(userid), len(itemid)

    user2idx = {v: i for i, v in enumerate(userid)}
    item2idx = {str(v): i for i, v in enumerate(itemid)}

    return user2idx, item2idx, n_user, m_item


def separate_data(train, test):
    valid = train.groupby('userID').tail(1)
    train = train.drop(index=valid.index)
    valid = valid.reset_index(drop=True)

    test = test[test.answerCode == -1]

    return train, valid, test


def process_data(train_data, valid_data, test_data, user2idx, item2idx, n_user, m_item, device):
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32) 
    train_edges = []; valid_edges = []; test_edges = []
    valid_label = []

    pos_edges = [[] for _ in range(n_user)]
    neg_edges = [[] for _ in range(n_user)]

    # generate edges 
    for user, item, acode in zip(train_data.userID, train_data.assessmentItemID_c, train_data.answerCode):
        uid, iid = user2idx[user], item2idx[str(item)]
        if acode == 1:
            pos_edges[uid].append(iid)
            train_mat[uid, iid] = 1.0
        else:
            neg_edges[uid].append(iid)
        train_edges.append([uid, iid])

    for user, item, acode in zip(valid_data.userID, valid_data.assessmentItemID_c, valid_data.answerCode):
        uid, iid = user2idx[user], item2idx[str(item)]
        valid_edges.append([uid, iid])
        valid_label.append(acode)

    for user, item, acode in zip(test_data.userID, test_data.assessmentItemID_c, test_data.answerCode):
        uid, iid = user2idx[user], item2idx[str(item)]
        test_edges.append([uid, iid])

    
    # construct degree matrix for graphmf
    items_D = np.sum(train_mat, axis=0).reshape(-1)
    users_D = np.sum(train_mat, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    
    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
    
    return pos_edges, neg_edges, train_edges, valid_edges, valid_label, test_edges, train_mat, constraint_mat


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