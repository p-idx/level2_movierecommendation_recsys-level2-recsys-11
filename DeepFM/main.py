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


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    set_seed(args.seed)
    check_path(args.output_dir)

    # Preprocess
    joined_rating_df = pd.read_csv(os.path.join(args.data_path, 'joined_df.csv'))
    data, field_dims, idx_dict = preprocess(joined_rating_df)

    # Dataloader
    train_loader, valid_loader = data_loader(args, data)

    # Train

    if 'model.pt' in os.listdir('./output'):
        model_path = os.path.join(args.output_dir, "model.pt")
        model = DeepFM(field_dims, args.embedding_dim, args.mlp_dims).to(device)
        # model = DeepFM(field_dims, args.embedding_dim, args.mlp_dims) # for debugging
        model.load_state_dict(torch.load(model_path))
        
    else:
        model = train(args, device, field_dims, train_loader, valid_loader)
        
    model.eval()
    # Inference
    print('make inference data...')
    LAST_USER_ID = 31359
    slice_num = 2_300
    user_list = [i * slice_num for i in range(1, LAST_USER_ID//slice_num)]
    print(user_list)
    user_list.append(LAST_USER_ID)
    slice_start = 0
    predict_output = dict()
    for slice_end in user_list[:1]: # user 잘라서 넣어야 함
        
        temp_list = [i for i in range(slice_start, slice_end)]
        sliced_df = joined_rating_df.query('user in @temp_list')
        inference_rating_df = make_inference_data(sliced_df, args)
        inference_data = mapping(inference_rating_df, idx_dict, args)
        predict_data = model(inference_data)
        
        slice_start = slice_end
        print(f'{(user_list.index(slice_end)+1) / (len(user_list)-1):.2f}% done')

    print(predict_data.shape)
if __name__ == "__main__":
    main()