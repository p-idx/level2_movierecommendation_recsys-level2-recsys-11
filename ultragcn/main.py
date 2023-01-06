import pandas as pd
import torch
import wandb
import os
import datetime

from args import parse_args
from src.dataset import prepare_dataset
from src.model import UltraGCN
from src.trainer import train, inference
from src.utils import setSeeds, check_path, get_local_time


def main():
    args = parse_args()
    device = torch.device("cuda" if args.use_cuda_if_available else "cpu")
    args.device = device 
    print(f"Device: {device}")

    setSeeds()
    check_path(args.model_save_path)

    if args.wandb and not args.inference:
        wandb.login()
        cur = get_local_time()
        wandb.init(
            project='movie_ultragcn',
            name=f'emb_dim={args.embedding_dim}, K={args.ii_neighbor_num}, gamma={args.GAMMA}, lambda={args.LAMBDA} | ' + cur
        )
        wandb.config.update(args)
    
    
    print('1. Loading Dataset...')
    constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, valid_loader, \
    interacted_items, test_ground_truth_list, mask, n_user, m_item, idx_dict = prepare_dataset(args)
    
    args.user_num, args.item_num = n_user, m_item
    print('Load Dataset Done')

    model = UltraGCN(args, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.inference:
        model_path = os.path.join(args.model_save_path, 'best_model.pt')

        if not os.path.exists(model_path):
            print('INFERENCE FAILED: NO MODEL FOUND')
            
        else:
            model.load_state_dict(torch.load(model_path))
    
            inference(args, model, n_user, mask, idx_dict)
        return
    
    else:
        print('2. Training...')
        train(
            model, 
            optimizer, 
            train_loader, 
            valid_loader, 
            mask,
            test_ground_truth_list, 
            interacted_items,
            args
        )

        print('Training Done!')
        wandb.finish()

if __name__ == "__main__":  
    main()
