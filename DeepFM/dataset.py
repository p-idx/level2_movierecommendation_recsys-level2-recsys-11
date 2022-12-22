import os
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

def preprocess(args):
    joined_rating_df = pd.read_csv('../data/train/joined_df.csv')
    print('df shape',joined_rating_df.shape)
    # genre, writer, director, year, title index mapping

    # TODO: 각 dict별로 offset 추가
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
    users = joined_rating_df.loc[:,'user'].unique()
    users.sort()
    items =  joined_rating_df.loc[:, 'item'].unique()
    items.sort()

    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
        users = joined_rating_df.loc[:,'user'].unique()
        
    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
        items =  joined_rating_df.loc[:, 'item'].unique()

    joined_rating_df = joined_rating_df.sort_values(by=['user'])
    joined_rating_df.reset_index(drop=True, inplace=True)

    data = joined_rating_df
    field_dims = np.array([
        len(users), len(items), len(genre_dict), len(writer_dict), len(director_dict),
                            len(year_dict), len(title_dict)
                            ], dtype=np.int32)

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
    
    offset = 0
    for col in tqdm(data.columns):
        data[col] = data[col] + offset
        offset += data[col].nunique()

    X = torch.tensor(data.drop('rating').values)
    y = torch.tensor(list(data.loc[:,'rating']))


    dataset = RatingDataset(X, y)
    
    # train, test split
    train_size = int(args.train_ratio * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader