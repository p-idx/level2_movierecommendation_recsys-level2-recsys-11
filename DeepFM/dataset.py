import os
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm

from config import CFG


def negative_sampling(raw_rating_df, num_negative):
    user_group_dfs = list(raw_rating_df.groupby('user')['item'])
    first_row = True
    user_neg_dfs = pd.DataFrame()

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(list(item - u_items), num_negative, replace=False)

        i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis=0, sort=False)
    
    raw_rating_df = pd.concat(raw_rating_df, user_neg_dfs, axis=0, sort=False)

    return raw_rating_df


def preprocess(CFG):
    print('Preprocessing...')

    # rating_df 생성
    rating_path = os.path.join(CFG[datapath], 'train', 'train_ratings.csv')
    raw_rating_df = pd.read_csv(rating_path)
    
    # implicit feedback
    raw_rating_df['rating'] = 1.0
    raw_rating_df.drop(['time'], axis=1, inplace=True)
    print('Creating negative instances...')
    raw_rating_df = negative_sampling(raw_rating_df, CFG['num_negative'])

    users = set(raw_rating_df.loc[:, 'user'])
    items = set(raw_rating_df.loc[:, 'item'])

    # genre_df 생성
    genre_path = os.path.join(CFG[datapath], 'train', 'genres.tsv')

    raw_genre_df = pd.read_csv(genre_path)
    raw_genre_df = raw_genre_df.drop_duplicates(subset=['item'])

    genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}
    raw_genre_df['genre'] = raw_genre_df['genre'].map(lambda x: genre_dict[x])

    # writers_df 생성
    writer_path = os.path.join(CFG[datapath], 'train', 'writers.tsv')

    raw_writer_df = pd.read_csv(writer_path)
    raw_writer_df = raw_writer_df.drop_duplicates(subset=['item'])

    writer_dict = {writer:i for i, writer in enumerate(set(raw_writer_df['writer']))}
    raw_writer_df['writer'] = raw_writer_df['writer'].map(lambda x: writer_dict[x])

    # directors_df 생성
    director_path = os.path.join(CFG[datapath], 'train', 'directors.tsv')

    raw_director_df = pd.read_csv(director_path)

    director_dict = {director:i for i, director in enumerate(set(raw_director_df['director']))}
    raw_director_df['director'] = raw_director_df['director'].map(lambda x: director_dict[x])


    # years_df 
    year_path = os.path.join(CFG[datapath], 'train', 'directors.tsv')

    raw_year_df = pd.read_csv(year_path)
    
    year_dict = {year:i for i, year in enumerate(set(raw_year_df['year']))}
    raw_year_df['year'] = raw_year_df['year'].map(lambda x: year_dict[x])


    # titles_df
    title_path = os.path.join(CFG[datapath], 'train', 'directors.tsv')

    raw_title_df = pd.read_csv(title_path)
    
    title_dict = {title:i for i, title in enumerate(set(raw_title_df['title']))}
    raw_title_df['title'] = raw_title_df['title'].map(lambda x: title_dict[x])
    
    # join dfs
    df_list = [raw_rating_df, raw_director_df, raw_genre_df, raw_title_df, raw_writer_df, raw_year_df]
    joined_rating_df = reduce(lambda  left,right: pd.merge(left,right,on=['DATE'],
                                            how='outer'), df_list).fillna()

    # user, item을 zero-based index로 mapping
    users = list(set(joined_rating_df.loc[:,'user']))
    users.sort()
    items =  list(set((joined_rating_df.loc[:, 'item'])))
    items.sort()
    genres =  list(set((joined_rating_df.loc[:, 'genre'])))
    genres.sort()

    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
        users = list(set(joined_rating_df.loc[:,'user']))
        
    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
        items =  list(set((joined_rating_df.loc[:, 'item'])))

    joined_rating_df = joined_rating_df.sort_values(by=['user'])
    joined_rating_df.reset_index(drop=True, inplace=True)

    data = joined_rating_df
    field_dims = np.array(len(users), len(items), len(genre_dict), len(writer_dict), len(director_dict),
                            len(year_dict), len(title_dict), type=np.uint32)

    print('Preprocess Done!')
    
    return data, field_dims, users, items


def data_loader():
    pass