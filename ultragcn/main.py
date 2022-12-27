import pandas as pd
import torch
from config import CFG
from src.datasets import prepare_dataset
from src.model import UltraGCN
from src.trainer import train
from src.utils import class2dict


# for specific gpu index,
# device = torch.device(f'cuda:{CFG.gpu}' if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    params = class2dict(CFG)
    use_cuda = torch.cuda.is_available() and params['use_cuda_if_available']
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    
    print('1. Loading Dataset...')
    constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, valid_loader, pos_edges, neg_edges, valid_label, params = prepare_dataset(device, params)
    print('Load Dataset Done')

    model = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train(
        model, 
        optimizer, 
        train_loader, 
        valid_loader, 
        valid_label,
        pos_edges, 
        neg_edges,
        params,
        device
    )
    print('END')
