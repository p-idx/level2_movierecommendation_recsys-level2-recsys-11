import os
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset


def negative_sampling(raw_rating_df, items, num_negative):
    user_group_dfs = list(raw_rating_df.groupby('user')['item'])
    first_row = True
    user_neg_dfs = pd.DataFrame()

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)

        i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis=0, sort=False)
    
    raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis=0, sort=False)

    return raw_rating_df


def preprocess(args):
    joined_rating_df = pd.read_csv('../data/train/joined_df')

    # genre, writer, director, year, title index mapping
    genre_dict = {genre:i for i, genre in enumerate(joined_rating_df['genre'].unique())}
    joined_rating_df['genre'] = joined_rating_df['genre'].map(genre_dict)

    writer_dict = {writer:i for i, writer in enumerate(joined_rating_df['writer'].unique())}
    joined_rating_df['writer'] = joined_rating_df['writer'].map(writer_dict)

    director_dict = {director:i for i, director in enumerate(joined_rating_df['director'].unique())}
    joined_rating_df['director'] = joined_rating_df['director'].map(director_dict)

    year_dict = {year:i for i, year in enumerate(joined_rating_df['year'].unique())}
    joined_rating_df['year'] = joined_rating_df['year'].map(year_dict)

    title_dict = {title:i for i, title in enumerate(joined_rating_df['title'].unique())}
    joined_rating_df['title'] = joined_rating_df['title'].map(lambda x: title_dict[x])

    # user, item을 zero-based index로 mapping
    users = list(set(joined_rating_df.loc[:,'user']))
    users.sort()
    items =  list(set((joined_rating_df.loc[:, 'item'])))
    items.sort()

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


# data loader 생성
class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor.long()
        self.target_tensor = target_tensor.long()

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)


# feature matrix X, label tensor y 생성
def data_loader(args, data, field_dims):
    user_col = torch.tensor(data.loc[:,'user'])
    item_col = torch.tensor(data.loc[:,'item'])
    genre_col = torch.tensor(data.loc[:,'genre'])
    writer_col = torch.tensor(data.loc[:, 'writer'])
    director_col = torch.tensor(data.loc[:, 'director'])
    year_col = torch.tensor(data.loc[:, 'year'])
    title_col = torch.tensor(data.loc[:, 'title'])


    offsets = [0] + np.cumsum(field_dims, axis=0).tolist()
    for col, offset in zip([user_col, item_col, genre_col], offsets):
        col += offset

    X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1),
                    writer_col.unsqueeze(1), director_col.unsqueeze(1), year_col.unsqueeze(1), 
                    title_col.unsqueeze(1)], dim=1)
    y = torch.tensor(list(data.loc[:,'rating']))


    dataset = RatingDataset(X, y)
    
    # train, test split
    train_size = int(args.train_ratio * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader