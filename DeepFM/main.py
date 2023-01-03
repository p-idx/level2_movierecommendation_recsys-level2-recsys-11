import time
import torch
from tqdm import tqdm

import pandas as pd
from args import parse_args
from dataset import preprocess, data_loader, make_inference_data, mapping
from model import DeepFM
from trainer import train
from utils import set_seed, check_path
import os
from collections import defaultdict


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = 'cpu'

    set_seed(args.seed)
    check_path(args.output_dir)

    # Preprocess
    joined_rating_df = pd.read_csv(os.path.join(args.data_path, 'joined_df.csv'))
    
    data, field_dims, idx_dict = preprocess(joined_rating_df)

    # Dataloader
    train_loader, valid_loader = data_loader(args, data)

    # Train
    if args.train:
        model = train(args, device, field_dims, train_loader, valid_loader)
    else:
        model_path = os.path.join(args.output_dir, "model.pt")
        model = DeepFM(field_dims, args.embedding_dim, args.mlp_dims, args).to('cpu')
        # model = DeepFM(field_dims, args.embedding_dim, args.mlp_dims) # for debugging
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    # Inference
    print('make inference data...')
    LAST_USER_ID = 31359
    slice_num = 500
    user_list = [i * slice_num for i in range(1, LAST_USER_ID//slice_num)]
    user_list.append(LAST_USER_ID)
    print(user_list)
    slice_start = 0
    predict_output = defaultdict(list)
    ITEM = 1
    USER = 0
    SCORE = 2
    joined_rating_df['user'] = joined_rating_df['user'].map(idx_dict['user2idx'])
    for slice_end in user_list: # user 잘라서 넣어야 함
        temp_list = [i for i in range(slice_start, slice_end+1)]
        sliced_df = joined_rating_df.query('user in @temp_list')
        # print(sliced_df)
        inference_rating_df = make_inference_data(sliced_df, args)
        inference_data = mapping(inference_rating_df, idx_dict, args)
        print('inference_data shape', inference_data.shape)
        predict_data = model(inference_data)
        predict_data = torch.cat([inference_data[:,0:2], predict_data.unsqueeze(1)], dim=1) # make top_k list
        # predict_data.int()
        for user in temp_list:
            temp_data = predict_data[predict_data[:,USER] == user][:,SCORE]
            topk = torch.topk(temp_data, 10, sorted=True)
            # predict_output[idx_dict['idx2user'][user]] = predict_data[topk.indices, ITEM]
            for item in predict_data[topk.indices, ITEM]:
                predict_output[idx_dict['idx2user'][user]].append(idx_dict['idx2item'][int(item)])
        slice_start = slice_end+1
        print(f'{((user_list.index(slice_end)+1) / (len(user_list))) * 100:.2f}% done')

        print(predict_data.shape)
    pd.DataFrame(predict_output).to_csv(f'test_{slice_end}.csv', index=False)
    # [138459, 138461, 138470, 138471, 138472, 138473, 138475, 138486, 138492, 138493]
if __name__ == "__main__":
    main()