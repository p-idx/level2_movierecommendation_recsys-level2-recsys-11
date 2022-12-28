import pandas as pd
import os
import re
from tqdm import tqdm

train_df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
year_data = pd.read_csv('/opt/ml/input/data/train/years.tsv', sep='\t')
genre_data = pd.read_csv('/opt/ml/input/data/train/genres.tsv', sep='\t')
title_data = pd.read_csv('/opt/ml/input/data/train/titles.tsv', sep='\t')
userid, itemid = train_df['user'].unique(), train_df['item'].unique()

userid_2_index = {v:i for i,v in enumerate(userid)}
itemid_2_index = {v:i for i,v in enumerate(itemid)}
index_2_itemid = {v:i for i,v in itemid_2_index.items()}
index_2_userid = {v:i for i,v in userid_2_index.items()}

path = f"dataset/movie"
interfile = f"dataset/movie/movie.inter"
userfile = f"dataset/movie/movie.user"
itemfile = f"dataset/movie/movie.item"
os.makedirs(path, exist_ok=True)

print("process .inter")
table = []
for user, item in tqdm(zip(train_df.user, train_df.item)):
    uid, iid = userid_2_index[user], itemid_2_index[item]
    table.append([uid, iid, 1])

with open(interfile, "w") as f:
    f.write("user:token\titem:token\trating:float\n")
    for row in table:
        f.write("\t".join([str(x) for x in row])+"\n")

print("process .user")
table = []
for i in userid_2_index.values():
    table.append([i])

with open(userfile, "w") as f:
    f.write("user:token\n")
    for row in table:
        f.write("\t".join([str(x) for x in row])+"\n")

print("process .item")
print("missing value preprocessing...")
year_merge = pd.merge(train_df, year_data, on='item', how='left')
year_merge = pd.merge(year_merge, title_data, on='item', how='left')
year_merge = year_merge.drop_duplicates(subset=['item'])
year_na = year_merge[year_merge.year.isna()]
year_merge.loc[year_merge.year.isna(),'year'] = year_na['title'].apply(lambda x : x[-5:-1])
year_merge.index = year_merge.item
year_merge.drop(['item','user','time'], axis=1, inplace=True)
year_merge['title'] = year_merge['title'].apply(lambda x : x[:-7])
year_merge['title'] = year_merge['title'].apply(lambda x : re.sub(r"[^0-9a-zA-Z\s]", "", x))
genre_agg = genre_data.groupby('item').agg(list)

table = []
for i in itemid_2_index.values():
    item_id = index_2_itemid[i]
    table.append([i, year_merge.loc[item_id,'title'], year_merge.loc[item_id,'year'], " ".join(genre_agg.loc[item_id][0])]) #  

with open(itemfile, "w") as f:
    f.write("item:token\ttitle:token_seq\tyear:token\tgenre:token_seq\n") # 
    for row in table:
        f.write("\t".join([str(x) for x in row])+"\n")