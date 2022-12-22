import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from functools import reduce
from utils import set_seed
from args import parse_args


def negative_sampling(raw_rating_df, items, args):
    print('negative sapmling..')
    user_group_dfs = list(raw_rating_df.groupby('user')['item'])
    user_neg_dfs = pd.DataFrame()
    temp_dict = {
        'user':[],
        'item':[]
    }

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        n_u_items = len(u_items)
        
        if n_u_items >= args.negative_threshold:
            num_negative = n_u_items * args.ratio_negative_long
        else:
            num_negative = n_u_items * args.ratio_negative
        i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)

        temp_dict['user'].extend([u]*len(i_user_neg_item))
        temp_dict['item'].extend(i_user_neg_item)
    
    user_neg_dfs = pd.DataFrame(temp_dict)
    raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis=0, sort=False)

    return raw_rating_df


def main():
    args = parse_args()
    set_seed(args.seed)

    print('Preprocessing...')

    # rating_df 생성
    rating_path = os.path.join(args.datapath, 'train_ratings.csv')
    raw_rating_df = pd.read_csv(rating_path)
    
    # implicit feedback
    raw_rating_df['rating'] = 1.0
    raw_rating_df.drop(['time'], axis=1, inplace=True)
    print('Creating negative instances...')

    items = set(raw_rating_df.loc[:, 'item'])
    raw_rating_df = negative_sampling(raw_rating_df, items, args)

    # genre_df 생성
    genre_path = os.path.join(args.datapath, 'genres.tsv')
    raw_genre_df = pd.read_csv(genre_path, sep='\t')
    raw_genre_df = raw_genre_df.drop_duplicates(subset=['item'])

    # writers_df 생성
    writer_path = os.path.join(args.datapath, 'writers.tsv')
    raw_writer_df = pd.read_csv(writer_path, sep='\t')
    raw_writer_df = raw_writer_df.drop_duplicates(subset=['item'])

    # directors_df 생성
    director_path = os.path.join(args.datapath, 'directors.tsv')
    raw_director_df = pd.read_csv(director_path, sep='\t')

    # years_df 
    year_path = os.path.join(args.datapath, 'years.tsv')
    raw_year_df = pd.read_csv(year_path, sep='\t')

    # titles_df
    title_path = os.path.join(args.datapath, 'titles.tsv')
    raw_title_df = pd.read_csv(title_path, sep='\t')
        
    # join dfs
    df_list = [raw_rating_df, raw_director_df, raw_genre_df, raw_title_df, raw_writer_df, raw_year_df]
    joined_rating_df = reduce(lambda  left,right: pd.merge(left,right,on='item',
                                            how='outer'), df_list).fillna()

    # save joined df
    joined_rating_df.to_csv('../data/train/joined_df', mode='w')

    print('joined_rating_df saved at ../data/train/')

if __name__ == "__main__":
    main()