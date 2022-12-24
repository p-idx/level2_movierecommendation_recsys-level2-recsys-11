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

def main(args):
    print(torch.cuda.is_available())

    model_name = args.model
    # config = Config(model='FISM', dataset="movie", config_file_list=['/opt/ml/input/mission/Recbole/movie.yaml'])
    config = Config(model=model_name, dataset="movie", config_file_list=['movie.yaml'])
    config['gpu_id'] = 'cuda:0'
    config['split_to'] = 20
    init_seed(config['seed'], config['reproducibility'])

    print(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if args.model == 'FISM':
        model = FISM(config, train_data.dataset).to(config['device'])
    elif args.model == 'EASE':
        model = EASE(config, train_data.dataset).to(config['device'])
    elif args.model == 'BPR':
        model = BPR(config, train_data.dataset).to(config['device'])
    elif args.model == 'ItemKNN':
        model = ItemKNN(config, train_data.dataset).to(config['device'])
    elif args.model == 'NeuMF':
        model = NeuMF(config, train_data.dataset).to(config['device'])
    elif args.model == 'ConvNCF':
        model = ConvNCF(config, train_data.dataset).to(config['device'])
    elif args.model == 'ConvNCFBPR':
        model = ConvNCFBPRLoss(config, train_data.dataset).to(config['device'])
    elif args.model == 'DMF':
        model = DMF(config, train_data.dataset).to(config['device'])
    elif args.model == 'NAIS':
        model = NAIS(config, train_data.dataset).to(config['device'])
    elif args.model == 'SpectralCF':
        model = SpectralCF(config, train_data.dataset).to(config['device'])
    elif args.model == 'GCMC':
        model = GCMC(config, train_data.dataset).to(config['device'])
    elif args.model == 'NGCF':
        model = NGCF(config, train_data.dataset).to(config['device'])
    elif args.model == 'LightGCN':
        model = LightGCN(config, train_data.dataset).to(config['device'])
    elif args.model == 'DGCF':
        model = DGCF(config, train_data.dataset).to(config['device'])
    elif args.model == 'LINE':
        model = LINE(config, train_data.dataset).to(config['device'])
    elif args.model == 'MultiVAE':
        model = MultiVAE(config, train_data.dataset).to(config['device'])
    elif args.model == 'MultiDAE':
        model = MultiDAE(config, train_data.dataset).to(config['device'])
    elif args.model == 'MacridVAE':
        model = MacridVAE(config, train_data.dataset).to(config['device'])
    elif args.model == 'CDAE':
        model = CDAE(config, train_data.dataset).to(config['device'])
    elif args.model == 'ENMF':
        model = ENMF(config, train_data.dataset).to(config['device'])
    elif args.model == 'NNCF':
        model = NNCF(config, train_data.dataset).to(config['device'])
    elif args.model == 'RaCT':
        model = RaCT(config, train_data.dataset).to(config['device'])
    elif args.model == 'RecVAE':
        model = RecVAE(config, train_data.dataset).to(config['device'])
    elif args.model == 'SLIMElasti':
        model = SLIMElastic(config, train_data.dataset).to(config['device'])
    elif args.model == 'SGL':
        model = SGL(config, train_data.dataset).to(config['device'])
    elif args.model == 'ADMMSLIM':
        model = ADMMSLIM(config, train_data.dataset).to(config['device'])
    elif args.model == 'NCEPLRec':
        model = NCEPLRec(config, train_data.dataset).to(config['device'])
    elif args.model == 'SimpleX':
        model = SimpleX(config, train_data.dataset).to(config['device'])
    elif args.model == 'NCL':
        model = NCL(config, train_data.dataset).to(config['device'])

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, verbose=True, show_progress=config['show_progress']
    )
    print(best_valid_result)
    print(best_valid_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='EASE')
    args = parser.parse_args()
    main(args)