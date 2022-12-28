import pandas as pd
import torch
from args import parse_args
from src.dataset import prepare_dataset
from src.model import UltraGCN
from src.trainer import train
from src.utils import setSeeds, check_path


if __name__ == "__main__":  

    args = parse_args()
    device = torch.device("cuda" if args.use_cuda_if_available else "cpu")
    args.device = device 
    print(f"Device: {device}")

    setSeeds()
    check_path(args.model_save_path)
    
    print('1. Loading Dataset...')
    constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, valid_loader, \
    interacted_items, test_ground_truth_list, mask, n_user, m_item = prepare_dataset(args)
    args.user_num, args.item_num = n_user, m_item
    print('Load Dataset Done')

    model = UltraGCN(args, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
