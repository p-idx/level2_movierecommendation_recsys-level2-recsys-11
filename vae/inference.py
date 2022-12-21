import os
import pandas as pd
import numpy as np
import bottleneck as bn
from scipy import sparse

from preprocess import numerize


def main():

    ## data load ##
    DATA_DIR = './data/train/'
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)
    # id 정리
    profile2id_for_infernece = dict((pid, i) for (i, pid) in enumerate(raw_data.user.unique()))
    show2id_for_inference = dict((sid,i) for i,sid in enumerate(raw_data['item'].unique()))

    # 원래 id로 돌려주기 위한 해쉬
    id2profile_for_infernece = dict((i, pid) for (i, pid) in enumerate(raw_data.user.unique()))
    id2show_for_inference = dict((i,sid) for i,sid in enumerate(raw_data['item'].unique()))

    ## ready data for inference ##
    raw_numerize = numerize(raw_data, profile2id_for_infernece, show2id_for_inference)

    # sparse matrix 만들기
    n_users = raw_numerize['uid'].max() + 1
    n_items = raw_numerize['sid'].nunique()

    rows, cols = raw_numerize['uid'], raw_numerize['sid']
    raw_numerize_data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, n_items))

    # 모델 불러오기
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    model