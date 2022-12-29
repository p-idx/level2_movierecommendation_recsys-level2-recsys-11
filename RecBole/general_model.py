import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import ast
import logging

from recbole.model.general_recommender.fism         import FISM
from recbole.model.general_recommender.ease         import EASE
from recbole.model.general_recommender.bpr          import BPR
from recbole.model.general_recommender.itemknn      import ItemKNN
from recbole.model.general_recommender.neumf        import NeuMF
from recbole.model.general_recommender.convncf      import ConvNCF
from recbole.model.general_recommender.convncf      import ConvNCFBPRLoss
from recbole.model.general_recommender.dmf          import DMF
from recbole.model.general_recommender.nais         import NAIS
from recbole.model.general_recommender.spectralcf   import SpectralCF
from recbole.model.general_recommender.gcmc         import GCMC
from recbole.model.general_recommender.ngcf         import NGCF
from recbole.model.general_recommender.lightgcn     import LightGCN
from recbole.model.general_recommender.dgcf         import DGCF
from recbole.model.general_recommender.line         import LINE
from recbole.model.general_recommender.multivae     import MultiVAE
from recbole.model.general_recommender.multidae     import MultiDAE
from recbole.model.general_recommender.macridvae    import MacridVAE
from recbole.model.general_recommender.cdae         import CDAE
from recbole.model.general_recommender.enmf         import ENMF
from recbole.model.general_recommender.nncf         import NNCF
from recbole.model.general_recommender.ract         import RaCT
from recbole.model.general_recommender.recvae       import RecVAE
from recbole.model.general_recommender.slimelastic  import SLIMElastic
from recbole.model.general_recommender.sgl          import SGL
from recbole.model.general_recommender.admmslim     import ADMMSLIM
from recbole.model.general_recommender.nceplrec     import NCEPLRec
from recbole.model.general_recommender.simplex      import SimpleX
from recbole.model.general_recommender.ncl          import NCL


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
    config['repeatable'] = False

    if args.model == 'EASE': #끝
        config['reg_weight'] = args.reg_weight
    elif args.model == 'FISM': #끝
        config['embedding_size'] = args.embedding_size
        config['alpha'] = args.alpha
        config['reg_weights'] = args.reg_weights
    elif args.model == 'BPR':
        config['embedding_size'] = args.embedding_size
    elif args.model == 'ItemKNN': # 끝
        config['k'] = args.k
        config['shrink'] = args.shrink
    elif args.model == 'NeuMF':
        config['mf_embedding_size'] = args.mf_embedding_size
        config['mlp_embedding_size'] = args.mlp_embedding_size
        config['mlp_hidden_size'] = args.mlp_hidden_size
        config['dropout_prob'] = args.dropout_prob
        config['mf_train'] = args.mf_train
        config['mlp_train'] = args.mlp_train
    elif args.model == 'ConvNCF':
        config['embedding_size'] = args.embedding_size
        config['cnn_channels'] = args.cnn_channels
        config['cnn_kernels'] = args.cnn_kernels
        config['cnn_strides'] = args.cnn_strides
        config['dropout_prob'] = args.dropout_prob
        config['reg_weights'] = args.reg_weights
    elif args.model == 'DMF':
        config['user_embedding_size'] = args.user_embedding_size
        config['item_embedding_size'] = args.item_embedding_size
        config['user_hidden_size_list'] = args.user_hidden_size_list
        config['item_hidden_size_list'] = args.item_hidden_size_list
        config['inter_matrix_type'] = args.inter_matrix_type
    elif args.model == 'NAIS':
        config['embedding_size'] = args.embedding_size
        config['weight_size'] = args.weight_size
        config['algorithm'] = args.algorithm
        config['alpha'] = args.alpha
        config['beta'] = args.beta
        config['reg_weights'] = args.reg_weights
    elif args.model == 'SpectralCF':
        config['embedding_size'] = args.embedding_size
        config['n_layers'] = args.n_layers
        config['reg_weight'] = args.reg_weight
    elif args.model == 'GCMC':
        config['accum'] = args.accum
        config['dropout_prob'] = args.dropout_prob
        config['gcn_output_dim'] = args.gcn_output_dim
        config['embedding_size'] = args.embedding_size
        config['sparse_feature'] = args.sparse_feature
        config['class_num'] = args.class_num
        config['num_basis_functions'] = args.num_basis_functions
    elif args.model == 'NGCF':
        config['embedding_size'] = args.embedding_size
        config['hidden_size_list'] = args.hidden_size_list
        config['node_dropout'] = args.node_dropout
        config['message_dropout'] = args.message_dropout
        config['reg_weight'] = args.reg_weight
    elif args.model == 'LightGCN':
        config['embedding_size'] = args.embedding_size
        config['n_layers'] = args.n_layers
        config['reg_weight'] = args.reg_weight
    elif args.model == 'DGCF':
        config['embedding_size'] = args.embedding_size
        config['n_factors'] = args.n_factors
        config['n_iterations'] = args.n_iterations
        config['n_layers'] = args.n_layers
        config['reg_weight'] = args.reg_weight
        config['cor_weight'] = args.cor_weight
    elif args.model == 'LINE':
        config['embedding_size'] = args.embedding_size
        config['order'] = args.order
        config['second_order_loss_weight'] = args.second_order_loss_weight
    elif args.model == 'MultiVAE':
        config['latent_dimension'] = args.latent_dimension
        config['mlp_hidden_size'] = args.mlp_hidden_size
        config['dropout_prob'] = args.dropout_prob
        config['anneal_cap'] = args.anneal_cap
        config['total_anneal_steps'] = args.total_anneal_steps
    elif args.model == 'MultiDAE':
        config['latent_dimension'] = args.latent_dimension
        config['mlp_hidden_size'] = args.mlp_hidden_size
        config['dropout_prob'] = args.dropout_prob
    elif args.model == 'MultiDAE':
        config['embedding_size'] = args.embedding_size
        config['dropout_prob'] = args.dropout_prob
        config['kfac'] = args.kfac
        config['nogb'] = args.nogb
        config['std'] = args.std
        config['encoder_hidden_size'] = args.encoder_hidden_size
        config['tau'] = args.tau
        config['anneal_cap'] = args.anneal_cap
        config['total_anneal_steps'] = args.total_anneal_steps
        config['reg_weights'] = args.reg_weights
    elif args.model == 'CDAE':
        config['loss_type'] = args.loss_type
        config['hid_activation'] = args.hid_activation
        config['out_activation'] = args.out_activation
        config['corruption_ratio'] = args.corruption_ratio
        config['embedding_size'] = args.embedding_size
        config['reg_weight_1'] = args.reg_weight_1
        config['reg_weight_2'] = args.reg_weight_2
    elif args.model == 'ENMF':
        config['dropout_prob'] = args.dropout_prob
        config['embedding_size'] = args.embedding_size
        config['reg_weight'] = args.reg_weight
        config['negative_weight'] = args.negative_weight
    elif args.model == 'NNCF':
        config['ui_embedding_size'] = args.ui_embedding_size
        config['neigh_embedding_size'] = args.neigh_embedding_size
        config['num_conv_kernel'] = args.num_conv_kernel
        config['conv_kernel_size'] = args.conv_kernel_size
        config['pool_kernel_size'] = args.pool_kernel_size
        config['mlp_hidden_size'] = args.mlp_hidden_size
        config['neigh_num'] = args.neigh_num
        config['dropout'] = args.dropout
        config['neigh_info_method'] = args.neigh_info_method
        config['resolution'] = args.resolution
    elif args.model == 'RaCT':
        config['latent_dimension'] = args.latent_dimension
        config['mlp_hidden_size'] = args.mlp_hidden_size
        config['dropout_prob'] = args.dropout_prob
        config['anneal_cap'] = args.anneal_cap
        config['total_anneal_steps'] = args.total_anneal_steps
        config['critic_layers'] = args.critic_layers
        config['metrics_k'] = args.metrics_k
        config['train_stage'] = args.train_stage
        config['pretrain_epochs'] = args.pretrain_epochs
        config['save_step'] = args.save_step
        config['pre_model_path'] = args.pre_model_path
    elif args.model == 'RecVAE':
        config['hidden_dimension'] = args.hidden_dimension
        config['latent_dimension'] = args.latent_dimension
        config['dropout_prob'] = args.dropout_prob
        config['beta'] = args.beta
        config['gamma'] = args.gamma
        config['mixture_weights'] = args.mixture_weights
        config['n_enc_epochs'] = args.n_enc_epochs
        config['n_dec_epochs'] = args.n_dec_epochs
    elif args.model == 'SLIMElastic':
        config['alpha'] = args.alpha
        config['l1_ratio'] = args.l1_ratio
        config['positive_only'] = args.positive_only
        config['hide_item'] = args.hide_item
    elif args.model == 'SGL':
        config['type'] = args.type
        config['n_layers'] = args.n_layers
        config['ssl_tau'] = args.ssl_tau
        config['embedding_size'] = args.embedding_size
        config['drop_ratio'] = args.drop_ratio
        config['reg_weight'] = args.reg_weight
        config['ssl_weight'] = args.ssl_weight
    elif args.model == 'ADMMSLIM':
        config['lambda1'] = args.lambda1
        config['lambda2'] = args.lambda2
        config['alpha'] = args.alpha
        config['rho'] = args.rho
        config['k'] = args.k
        config['positive_only'] = args.positive_only
        config['center_columns'] = args.center_columns
    elif args.model == 'NCEPLRec':
        config['rank'] = args.rank
        config['beta'] = args.beta
        config['reg_weight'] = args.reg_weight
    elif args.model == 'SimpleX':
        config['embedding_size'] = args.embedding_size
        config['margin'] = args.margin
        config['negative_weight'] = args.negative_weight
        config['gamma'] = args.gamma
        config['aggregator'] = args.aggregator
        config['history_len'] = args.history_len
        config['reg_weight'] = args.reg_weight
    elif args.model == 'NCL':
        config['embedding_size'] = args.embedding_size
        config['n_layers'] = args.n_layers
        config['reg_weight'] = args.reg_weight
        config['ssl_temp'] = args.ssl_temp
        config['ssl_reg'] = args.ssl_reg
        config['hyper_layers'] = args.hyper_layers
        config['alpha'] = args.alpha
        config['proto_reg'] = args.proto_reg
        config['num_clusters'] = args.num_clusters
        config['m_step'] = args.m_step
        config['warm_up_step'] = args.warm_up_step



def main(args):
    print(torch.cuda.is_available())

    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

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
    parser.add_argument('-m', '--model', type=str, default='EASE')
    parser.add_argument('--lr', type=float, default=1e-3)


    parser.add_argument('--reg_weight', type=float, default=50.0)
    parser.add_argument('--cor_weight', type=float, default=1e-2)
    parser.add_argument('--reg_weight_1', type=float, default=1.0)
    parser.add_argument('--reg_weight_2', type=float, default=0.01)
    parser.add_argument('--ssl_weight', type=float, default=0.05)
    parser.add_argument('--ssl_temp', type=float, default=0.1)
    parser.add_argument('--ssl_reg', type=float, default=1e-7)
    parser.add_argument('--negative_weight', type=float, default=0.5)
    parser.add_argument('--second_order_loss_weight', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--lambda1', type=float, default=3.0)
    parser.add_argument('--lambda2', type=float, default=200.0)
    parser.add_argument('--shrink', type=float, default=0.0)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--node_dropout', type=float, default=0.0)
    parser.add_argument('--anneal_cap', type=float, default=0.2)
    parser.add_argument('--message_dropout', type=float, default=0.1)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--resolution', type=float, default=0.01)
    parser.add_argument('--corruption_ratio', type=float, default=0.5)
    parser.add_argument('--l1_ratio', type=float, default=0.02)
    parser.add_argument('--ssl_tau', type=float, default=0.5)
    parser.add_argument('--drop_ratio', type=float, default=0.1)
    parser.add_argument('--rho', type=float, default=4000)
    parser.add_argument('--margin', type=float, default=0.9)
    parser.add_argument('--proto_reg', type=float, default=8e-8)

    parser.add_argument('--sparse_feature', type=arg_as_bool, default=True)
    parser.add_argument('--mf_train', type=arg_as_bool, default=True)
    parser.add_argument('--mlp_train', type=arg_as_bool, default=True)
    parser.add_argument('--nogb', type=arg_as_bool, default=True)
    parser.add_argument('--positive_only', type=arg_as_bool, default=True)
    parser.add_argument('--hide_item', type=arg_as_bool, default=True)
    parser.add_argument('--center_columns', type=arg_as_bool, default=False)

    parser.add_argument('--cnn_channels', type=arg_as_lst, default=[1, 32, 32, 32, 32])
    parser.add_argument('--cnn_kernels', type=arg_as_lst, default=[4,4,2,2])
    parser.add_argument('--cnn_strides', type=arg_as_lst, default=[4,4,2,2])
    parser.add_argument('--reg_weights', type=arg_as_lst, default=[1e-2, 1e-2])
    parser.add_argument('--encoder_hidden_size', type=arg_as_lst, default=[600])
    parser.add_argument('--user_hidden_size_list', type=arg_as_lst, default=[64, 64])
    parser.add_argument('--item_hidden_size_list', type=arg_as_lst, default=[64, 64])
    parser.add_argument('--hidden_size_list', type=arg_as_lst, default=[64, 64, 64])
    parser.add_argument('--critic_layers', type=arg_as_lst, default=[100,100,10])
    parser.add_argument('--mixture_weights', type=arg_as_lst, default=[0.15, 0.75, 0.1])
    parser.add_argument('--mlp_hidden_size', type=arg_as_lst, default=[128,64])

    parser.add_argument('--user_embedding_size', type=int, default=64)
    parser.add_argument('--item_embedding_size', type=int, default=64)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--mf_embedding_size', type=int, default=64)
    parser.add_argument('--mlp_embedding_size', type=int, default=64)
    parser.add_argument('--ui_embedding_size', type=int, default=64)
    parser.add_argument('--neigh_embedding_size', type=int, default=64)
    parser.add_argument('--num_conv_kernel', type=int, default=64)
    parser.add_argument('--conv_kernel_size', type=int, default=64)
    parser.add_argument('--neigh_num', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--hyper_layers', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=4)
    parser.add_argument('--n_factors', type=int, default=4)
    parser.add_argument('--gcn_output_dim', type=int, default=500)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--num_basis_functions', type=int, default=2)
    parser.add_argument('--latent_dimension', type=int, default=2)
    parser.add_argument('--total_anneal_steps', type=int, default=200000)
    parser.add_argument('--kfac', type=int, default=200000)
    parser.add_argument('--metrics_k', type=int, default=200000)
    parser.add_argument('--pretrain_epochs', type=int, default=150)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--hidden_dimension', type=int, default=10)
    parser.add_argument('--n_enc_epochs', type=int, default=3)
    parser.add_argument('--n_dec_epochs', type=int, default=1)
    parser.add_argument('--rank', type=int, default=450)
    parser.add_argument('--history_len', type=int, default=50)
    parser.add_argument('--num_clusters', type=int, default=1000)
    parser.add_argument('--m_step', type=int, default=1)
    parser.add_argument('--warm_up_step', type=int, default=20)

    parser.add_argument('--inter_matrix_type', type=str, default='01')
    parser.add_argument('--neigh_info_method', type=str, default='random')
    parser.add_argument('--loss_type', type=str, default='BCE')
    parser.add_argument('--hid_activation', type=str, default='relu')
    parser.add_argument('--out_activation', type=str, default='sigmoid')
    parser.add_argument('--weight_size', type=int, default=64)
    parser.add_argument('--algorithm', type=str, default='prod')
    parser.add_argument('--accum', type=str, default='01')
    parser.add_argument('--train_stage', type=str, default='actor_pretrain')
    parser.add_argument('--pre_model_path', type=str, default='')
    parser.add_argument('--type', type=str, default='ED')
    parser.add_argument('--aggregator', type=str, default='mean')


    args = parser.parse_args()
    main(args)