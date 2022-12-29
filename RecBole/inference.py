from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch
import gc
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from recbole.data import create_dataset, data_preparation
from pathlib import Path

#라벨 인코딩
train_df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
userid, itemid = train_df['user'].unique(), train_df['item'].unique()
index_2_userid = {i:v for i,v in enumerate(userid)}
index_2_itemid = {i:v for i,v in enumerate(itemid)}

# data_path = sorted(Path('./saved').iterdir(), key=os.path.getmtime)[-1].name
data_path = '/opt/ml/level2_movierecommendation_recsys-level2-recsys-11/RecBole/saved/DeepFM-Dec-28-2022_05-01-56.pth' # 그냥 특정 파일을 하고 싶을 때는 이걸 사용하세요
output_path = './output/'
file_name = output_path + data_path[73:-3] + 'csv'

print(f'load data and model from || {data_path}')
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(data_path)
del dataset
del train_data
del valid_data
del test_data

print(f'change config for submit')
config['eval_args']['split']['LS'] = 'test_only'
del config['eval_args']['split']['RS']
config['split_to'] = 20               # default = 0, cuda out of memory 때문에 늘리기
# config["eval_batch_size"] = 128        # cuda out of memory 해결 방법 : 배치사이즈를 낮춰봐라 (실패)
# config["train_batch_size"] = 128       # cuda out of memory 해결 방법 : 배치사이즈를 낮춰봐라 (실패)
print('recommend top 10')
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

del train_data
del valid_data
# gc.collect()
# torch.cuda.empty_cache()              # cuda out of memory 해결 방법 : cache 제거 (실패)
# test_data._dataset_kind = -1            # test_data 를 cpu로 할당하는 방법
model.to('cpu')         # mode을 cpu에 올림 : inference 할 때는 cuda out of memory 때문에 cpu에서 진행
a, b = full_sort_topk(np.arange(1, 10000), model.to('cpu'), test_data, 10, 'cpu') # 31361, device=config['device'] : device를 cuda로 하면 out of memory, cpu로 하면 cpu랑 cuda 동시사용 불가 에러
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
