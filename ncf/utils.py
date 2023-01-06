import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from tqdm import tqdm

from typing import Tuple

import pickle

class NegativeSampler:
    def __init__(self, ratings: pd.DataFrame, positives: pd.Series, freq_rate=0.2):
        self.ratings = ratings
        self.positives = positives
        self.positives_set = self.positives.apply(set)
        self.n_items = self.ratings['item'].nunique()
        
        all_movies = set([i for i in range(self.n_items)])

        try:
            with open('asset/negatives.pickle', 'rb') as fr:
                negatives = pickle.load(fr)
        except:
            negatives = []
            for u, pos_items_set in tqdm(self.positives_set.iteritems(), total=len(positives), desc='make negatives'):
                negatives.append(list(all_movies - pos_items_set))

            with open('asset/negatives.pickle', 'wb') as fw:
                pickle.dump(negatives, fw)

        # try:
        #     with open('asset/freq_negatives.pickle', 'rb') as fr:
        #         freq_negatives = pickle.load(fr)
        #     print('load freq negs')
        # except:
        #     freq_negatives = []
        #     for u, neg_items in tqdm(self.negatives.iteritems(), total=len(self.negatives), desc='make freq negs'):
        #         neg_freq = []
        #         for i in neg_items:
        #             neg_freq.extend([i for _ in range(self.freq[i])])
        #         freq_negatives.append(neg_freq)

        #     with open('asset/freq_negatives.pickle', 'wb') as fw:
        #         pickle.dump(freq_negatives, fw)

        # self.freq_negatives = pd.Series(freq_negatives)
            
        self.negatives = pd.Series(negatives)
        
        self.freq = (self.ratings['item'].value_counts() ** freq_rate).apply(int)
        self.freq_negatives = None


    def uniform_sampling(self, user, n_negs):
        # neg_items = []
        # for _ in range(n_negs):        
        #     neg_item = np.random.randint(len(self.negatives[user]))
        #     while neg_item in neg_items:
        #         neg_item = np.random.randint(len(self.negatives[user]))
        #     neg_items.append(neg_item)
            
        return random.choices(self.negatives[user], k=n_negs)


    def freq_sampling(self, user, n_negs):
        if self.freq_negatives == None:
            try:
                with open('asset/freq_negatives.pickle', 'rb') as fr:
                    freq_negatives = pickle.load(fr)
                print('load freq negs')
            except:
                freq_negatives = []
                for u, neg_items in tqdm(self.negatives.iteritems(), total=len(self.negatives), desc='make freq negs'):
                    neg_freq = []
                    for i in neg_items:
                        neg_freq.extend([i for _ in range(self.freq[i])])
                    freq_negatives.append(neg_freq)

                with open('asset/freq_negatives.pickle', 'wb') as fw:
                    pickle.dump(freq_negatives, fw)

            self.freq_negatives = pd.Series(freq_negatives)
        
        neg_items = []
        for _ in range(n_negs):        
            neg_item = np.random.randint(len(self.freq_negatives[user]))
            while neg_item in neg_items:
                neg_item = np.random.randint(len(self.freq_negatives[user]))
            neg_items.append(neg_item)
            
        return neg_items


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


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
    print(post_fix, flush=True)

    return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)


def get_sub_dataset(ratings: pd.DataFrame, n_valid) -> Tuple[pd.DataFrame, list]:
    sub_users = []
    sub_items = []
    sub_answers = []

    user_positives = ratings.groupby('user')['item'].apply(list)
    for u, items in tqdm(user_positives.iteritems(), total=len(user_positives), desc='get subset'):

        lasts = items[-1:]
        cp_items = items[:-1]
        
        pops = []
        for i in range(n_valid):
            pops.append(cp_items.pop(np.random.randint(len(cp_items))))
        sub_answers.append(lasts + pops)
        for i in cp_items:
            sub_users.append(u)
            sub_items.append(i)

    sub_ratings = pd.DataFrame(
        {
            'user': sub_users,
            'item': sub_items,
        }
    )
    return sub_ratings, sub_answers


def generate_submission_file(data_file, preds, name, time_info, k):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        f"output/{name}_{time_info}_k{k}_submission.csv", index=False
    )


## METRICS ##

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
