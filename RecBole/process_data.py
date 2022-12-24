import pandas as pd
import os
from tqdm import tqdm

train_df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
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
table = []
for i in itemid_2_index.values():
    table.append([i])

with open(itemfile, "w") as f:
    f.write("item:token\n")
    for row in table:
        f.write("\t".join([str(x) for x in row])+"\n")