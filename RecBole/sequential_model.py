import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import ast
import logging

from recbole.model.sequential_recommender.fpmc import FPMC
from recbole.model.sequential_recommender.sasrec import SASRec

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed#, set_color

from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import wandb

def set_config(args, config):
    config['gpu_id'] = 'cuda:0'
    config['split_to'] = 2
    init_seed(config['seed'], config['reproducibility'])
    config['learning_rate'] = args.lr
    config['log_wandb'] = True
    config['eval_args']['order'] = 'TO'
    config['repeatable'] = True
    config['show_progress'] = False

    if args.model == 'FPMC':
        config['loss_type'] = args.loss_type
        config['embedding_size'] = args.embedding_size


def main(args):
    print(torch.cuda.is_available())
    cur = datetime.datetime.now() + datetime.timedelta(hours=9)
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    # cur = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')
    config = Config(model=args.model, dataset="movie", config_file_list=['movie.yaml'])
    set_config(args, config)
    init_logger(config)
    wandb.init(
        project=f'kdg_{args.model}',
        name=f'{args.model}_{cur}',
        config=args,
    )
    logging.info(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = eval(args.model)(config, train_data.dataset).to(config['device'])

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, verbose=True, show_progress=config['show_progress']
    )
    print(best_valid_score)
    wandb.log(best_valid_result)
    wandb.finish()

if __name__ == '__main__':

    def arg_as_lst(s):
        v = ast.literal_eval(s)
        if not isinstance(v, list):
            raise argparse.ArgumentTypeError("not a list")
        return v
    def arg_as_bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='SASRec')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--dropout_prob', type=float, default=0.2)
    parser.add_argument('--mlp_hidden_size', type=arg_as_lst, default=[128,128,128])
    parser.add_argument('--loss_type', type=str, default='BPR')
    parser.add_argument('--embedding_size', type=int, default=10)

    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--inner_size', type=int, default=256)
    parser.add_argument('--hidden_dropoup_prob', type=float, default=0.5)
    parser.add_argument('--attn_dropout_prob', type=float, default=0.5)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    
    

    args = parser.parse_args()
    main(args)