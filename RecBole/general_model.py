import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

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

def set_config(args, config):
    config['gpu_id'] = 'cuda:0'
    config['split_to'] = 2
    init_seed(config['seed'], config['reproducibility'])
    config['learning_rate'] = args.lr

    if args.model == 'EASE':
        config['reg_weight'] = eval('args.' + str(args.model) + '_reg_weight')
    elif args.model == 'FISM':
        config['embedding_size'] = eval('args.' + str(args.model) + '_embedding_size')
        config['alpha'] = eval('args.' + str(args.model) + '_alpha')
        config['reg_weights'] = eval('args.' + str(args.model) + '_reg_weights')
    elif args.model == 'MultiVAE':
        config["mlp_hidden_size"] = eval('args.' + str(args.model) + '_mlp_hidden_size')
        config["latent_dimension"] = eval('args.' + str(args.model) + '_latent_dimension')
        config["dropout_prob"] = eval('args.' + str(args.model) + '_dropout_prob')
        config["anneal_cap"] = eval('args.' + str(args.model) + '_anneal_cap')
        config["total_anneal_steps"] = eval('args.' + str(args.model) + '_total_anneal_steps')

def main(args):
    print(torch.cuda.is_available())

    config = Config(model=args.model, dataset="movie", config_file_list=['movie.yaml'])
    set_config(args, config)

    print(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = eval(args.model)(config, train_data.dataset).to(config['device'])

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, verbose=True, show_progress=config['show_progress']
    )
    print(best_valid_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='EASE')
    parser.add_argument('--lr', type=float, default=1e-3)

    #EASE
    parser.add_argument('--EASE_reg_weight', type=float, default='250.0')

    #FISM
    parser.add_argument('--FISM_embedding_size', type=int, default='64')
    parser.add_argument('--FISM_alpha', type=float, default='0.0')
    parser.add_argument('--FISM_reg_weights', type=float, nargs='+', default=[1e-2, 1e-2])

    parser.add_argument('--ItemKNN_k', type=int, default='100')
    parser.add_argument('--ItemKNN_shrink', type=float, default='0.0')

    parser.add_argument('--MultiVAE_latent_dimension', type=int, default='128')
    parser.add_argument('--MultiVAE_mlp_hidden_size', type=list, default=[600])
    parser.add_argument('--MultiVAE_dropout_prob', type=float, default='0.5')
    parser.add_argument('--MultiVAE_anneal_cap', type=float, default='0.2')
    parser.add_argument('--MultiVAE_total_anneal_steps', type=int, default='200000')


    args = parser.parse_args()
    main(args)