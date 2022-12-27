import os
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

def preprocess(joined_rating_df):
    joined_rating_df = joined_rating_df.copy()
    
    print('DATA PREPROCESSING...')
    # user, item을 index로 mapping

    offset = 0 # 0은 NA
    user2idx = {user:idx for idx, user in enumerate(joined_rating_df['user'].unique(), offset)}
    idx2user = {idx:user for user, idx in user2idx.items()}
    joined_rating_df['user'] = joined_rating_df['user'].map(user2idx)

    offset += len(user2idx)
    item2idx = {item:idx for idx, item in enumerate(joined_rating_df['item'].unique(), offset)}
    idx2item = {idx:item for item, idx in item2idx.items()}
    joined_rating_df['item'] = joined_rating_df['item'].map(item2idx)

    # genre, writer, director, year, title index mapping

    offset += len(item2idx)
    genre2idx = {genre:i for i, genre in enumerate(joined_rating_df['genre'].unique(), offset)}
    joined_rating_df['genre'] = joined_rating_df['genre'].map(genre2idx)

    offset += len(genre2idx)    
    writer2idx = {writer:i for i, writer in enumerate(joined_rating_df['writer'].unique(), offset)}
    joined_rating_df['writer'] = joined_rating_df['writer'].map(writer2idx)

    offset += len(writer2idx)
    director2idx = {director:i for i, director in enumerate(joined_rating_df['director'].unique(), offset)}
    joined_rating_df['director'] = joined_rating_df['director'].map(director2idx)
    
    offset += len(director2idx)
    year2idx = {year:i for i, year in enumerate(joined_rating_df['year'].unique(), offset)}
    joined_rating_df['year'] = joined_rating_df['year'].map(year2idx)

    offset += len(year2idx)
    title2idx = {title:i for i, title in enumerate(joined_rating_df['title'].unique(), offset)}
    joined_rating_df['title'] = joined_rating_df['title'].map(title2idx)

    idx_dict = {
                'user2idx': user2idx,
                'idx2user': idx2user,
                'item2idx': item2idx,
                'idx2item': idx2item,
                'genre2idx': genre2idx,
                'writer2idx': writer2idx,
                'director2idx': director2idx,
                'year2idx': year2idx,
                'title2idx': title2idx
                }

    joined_rating_df = joined_rating_df.sort_values(by=['user'])
    joined_rating_df.reset_index(drop=True, inplace=True)

    data = joined_rating_df
    field_dims = np.array([len(user2idx), len(item2idx), len(genre2idx), len(writer2idx), len(director2idx),
                            len(year2idx), len(title2idx)], dtype=np.uint32)

    print('PREPROCESSING DONE!')
    
    return data, field_dims, idx_dict


# data loader 생성
class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)


# feature matrix X, label tensor y 생성
def data_loader(args, data):
    
    # offset = 0
    # for col in tqdm(data.columns):
    #     data[col] = data[col] + offset
    #     offset += data[col].nunique()

    data_x = data.drop('rating', axis=1)
    data_y = data['rating']

    X = torch.LongTensor(data_x.values)
    y = torch.LongTensor(data_y.values)
    dataset = RatingDataset(X, y)
    
    # train, test split
    train_size = int(args.train_ratio * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return train_loader, test_loader


def make_inference_data(
    joined_rating_df:pd.DataFrame,
    args
    ) -> pd.DataFrame:
    # 3. infrence instance 생성

    user_group_dfs = list(joined_rating_df.groupby('user')['item'])
    items=set(joined_rating_df['item'].unique())
    df_dict= {
            'user':[],
            'item':[]
            }
    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = list(items - u_items)
        
        df_dict['user'].extend([u]*len(i_user_neg_item))
        df_dict['item'].extend(i_user_neg_item)
    # inference_file = 'inference_raw_df.csv'
    # inference_file_path = os.path.join(args.data_path, inference_file)
    # if inference_file in os.listdir(args.data_path):
    #     print('you have inference file')
    #     inference_rating_df = pd.read_csv(inference_file_path)
    # else:
    #     print("you don't have inference file")
    #     inference_rating_df = pd.DataFrame(df_dict)
    #     inference_rating_df.to_csv(inference_file_path, index=False)
    inference_rating_df = pd.DataFrame(df_dict)

    return inference_rating_df

def mapping(
    inference_rating_df:pd.DataFrame, 
    idx_dict:dict,
    args
    ) -> torch.Tensor:

    raw_genre_data = os.path.join(args.data_path, "genres.tsv")
    
    raw_writer_data = os.path.join(args.data_path, "writers.tsv")
    
    raw_director_data = os.path.join(args.data_path, "directors.tsv")
    
    raw_year_data = os.path.join(args.data_path, "years.tsv")
    raw_title_data = os.path.join(args.data_path, "titles.tsv")

    raw_genre_df = pd.read_csv(raw_genre_data, sep='\t')
    raw_genre_df = raw_genre_df.drop_duplicates(subset=['item'])
    raw_writer_df = pd.read_csv(raw_writer_data, sep='\t')
    raw_writer_df = raw_writer_df.drop_duplicates(subset=['item'])
    raw_director_df = pd.read_csv(raw_director_data, sep='\t')     
    raw_director_df = raw_director_df.drop_duplicates(subset=['item'])
    raw_year_df = pd.read_csv(raw_year_data, sep='\t')
    raw_title_df = pd.read_csv(raw_title_data, sep='\t')

    print('merge start')
    joined_inference_df = pd.merge(inference_rating_df, raw_genre_df, left_on='item', right_on='item', how='left')
    del inference_rating_df, raw_genre_df
    joined_inference_df = pd.merge(joined_inference_df, raw_writer_df, left_on='item', right_on='item', how='left')
    del raw_writer_df
    joined_inference_df = pd.merge(joined_inference_df, raw_year_df, left_on='item', right_on='item', how='left')
    del raw_year_df
    joined_inference_df = pd.merge(joined_inference_df, raw_director_df, left_on='item', right_on='item', how='left')
    del raw_director_df
    joined_inference_df = pd.merge(joined_inference_df, raw_title_df, left_on='item', right_on='item', how='left')
    del raw_title_df
    # joined_inference_df = joined_inference_df.fillna(0)
    print('merge done')


    joined_inference_df['user'] = joined_inference_df['user'].map(idx_dict['user2idx'])
    joined_inference_df['item'] = joined_inference_df['item'].map(idx_dict['item2idx'])
    joined_inference_df['genre'] = joined_inference_df['genre'].map(idx_dict['genre2idx'])
    joined_inference_df['writer'] = joined_inference_df['writer'].map(idx_dict['writer2idx'])
    joined_inference_df['director'] = joined_inference_df['director'].map(idx_dict['director2idx'])
    joined_inference_df['year'] = joined_inference_df['year'].map(idx_dict['year2idx'])
    joined_inference_df['title'] = joined_inference_df['title'].map(idx_dict['title2idx'])
    # joined_inference_df = joined_inference_df.fillna(0) 
    # joined_inference_df = joined_inference_df.drop_duplicates(['user', 'item'])

    return torch.tensor(joined_inference_df.values).to('cuda').long()
    # return torch.tensor(joined_inference_df.values).long() # for debugging
    
def make_top_k_list() -> list:
    pass