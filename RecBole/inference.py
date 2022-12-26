from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from recbole.data import create_dataset, data_preparation
from pathlib import Path

#라벨 인코딩
train_df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
userid, itemid = train_df['user'].unique(), train_df['item'].unique()
index_2_userid = {i:v for i,v in enumerate(userid)}
index_2_itemid = {i:v for i,v in enumerate(itemid)}

# data_path = '/opt/ml/input/level2_movierecommendation_recsys-level2-recsys-11/RecBole/saved/EASE-Dec-24-2022_16-14-55.pth' # 그냥 특정 파일을 하고 싶을 때는 이걸 사용하세요
data_path = sorted(Path('./saved').iterdir(), key=os.path.getmtime)[-1].name
file_name = 'output/' + data_path[:-3] + 'csv'

print(f'load data and model from || {data_path}')
config, model, dataset, train_data, valid_data, test_data = load_data_and_model('saved/' + data_path)

print(f'change config for submit')
config['eval_args']['split']['LS'] = 'test_only'
del config['eval_args']['split']['RS']
config['split_to'] = 0


print('recommend top 10')
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
a, b = full_sort_topk(np.arange(1, 31361), model, test_data, 10)
c = b.tolist()

print('top 10 recommended. to csv')
answer = []
for i, j in tqdm(enumerate(c)):
    name = index_2_userid[i]
    for k in j:
        answer.append((name, index_2_itemid[k-1]))

dataframe = pd.DataFrame(answer, columns=["user", "item"])

if not os.path.exists('./output'):
    os.mkdir('./output')
print(f'file name is {file_name}')
dataframe.to_csv(file_name, index=False)
