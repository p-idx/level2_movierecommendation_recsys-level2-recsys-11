import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import bottleneck as bn
import random
import os

from scipy import sparse
from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
from dataset import DataLoader
from preprocess import numerize
from models import MultiDAE, MultiVAE, loss_function_dae, loss_function_vae

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def train(model, train_data, criterion, optimizer, is_VAE = False):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count
    
    N = train_data.shape[0]
    idxlist = list(range(N))
    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(args.device)
        optimizer.zero_grad()

        if args.model == 'VAE':
          if args.total_anneal_steps > 0:
              anneal = min(args.anneal_cap, 
                              1. * update_count / args.total_anneal_steps)
          else:
              anneal = args.anneal_cap

          #TODO
          optimizer.zero_grad()
          recon_batch, mu, logvar = model(data)
          #model에 입력 출력 코드를 작성해주세요

          #loss 함수를 설정해주세요
          loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
          recon_batch = model(data)
          loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        args.epochs, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            

            start_time = time.time()
            train_loss = 0.0


def evaluate(model, criterion, data_tr, data_te, is_VAE=False):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r10_list = []
    r20_list = []
    r50_list = []

    N = data_tr.shape[0]
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(args.device)
            if args.model == 'VAE' :
              if args.total_anneal_steps > 0:
                  anneal = min(args.anneal_cap, 
                                1. * update_count / args.total_anneal_steps)
              else:
                  anneal = args.anneal_cap
            
              #TODO
              #model에 입력 출력 코드를 작성해주세요
              recon_batch, mu, logvar = model(data_tensor)
              #loss 함수를 설정해주세요
              loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
              recon_batch = model(data_tensor)
              loss = criterion(recon_batch, data_tensor)




            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            r10_list.append(r10)
            r20_list.append(r20)
            r50_list.append(r50)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r10_list = np.concatenate(r10_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list), np.mean(r50_list)



## 각종 파라미터 세팅
def main(args):
    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    args.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')

    args.device = torch.device("cuda" if args.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    loader = DataLoader(args.data)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    train_data_tr, train_data_te = loader._load_tr_te_data('train')
    test_data_tr, test_data_te = loader.load_data('test')

    

    ###############################################################################
    # Build the model
    ###############################################################################

    p_dims = [200, 600, n_items]

    ###############################################################################
    # Training code
    ###############################################################################

    best_n100 = -np.inf
    update_count = 0
    if args.model == 'DAE':
        print('model : DAE, train start')
        model = MultiDAE(p_dims).to(args.device)
        criterion = loss_function_dae
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
        is_VAE = False
    else:
        print('model : VAE, train start')
        model = MultiVAE(p_dims).to(args.device)
        criterion = loss_function_vae
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
        is_VAE = True

    print(f'training for {args.epochs} epochs')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # update_count = 0
        train(model, train_data_tr, criterion, optimizer)
        val_loss, n100, r10, r20, r50 = evaluate(model, criterion, train_data_tr, train_data_te)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    n100, r10, r20, r50))
        print('-' * 89)

        # n_iter = args.epochs * len(range(0, N, args.batch_size))

        # Save the model if the n100 is the best we've seen so far.
        if n100 > best_n100:
            with open(args.model + args.save, 'wb') as f:
                torch.save(model, f)
            best_n100 = n100

    print(f'{args.model}.best model saved complete')

    print('using best model, check test(validation) loss')
    # Load the best saved model and check validation metric
    with open(args.model + args.save, 'rb') as f:
        model = torch.load(f)
    
    # Run on test data. (전혀 다른 유저들에 대한)
    test_loss, n100, r10, r20, r50 = evaluate(model, criterion, test_data_tr, test_data_te, is_VAE=False)
    print('=' * 89)
    print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r10 {:4.2f} | r20 {:4.2f} | '
            'r50 {:4.2f}'.format(test_loss, n100, r10, r20, r50))
    print('=' * 89)
    

    # inference start
    print('=' * 89)
    print('inference start')
    print('=' * 89)

    DATA_DIR = '../data/train/'
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)
    # id 정리
    profile2id_for_infernece = dict((pid, i) for (i, pid) in enumerate(raw_data.user.unique()))
    show2id_for_inference = dict((sid,i) for i,sid in enumerate(raw_data['item'].unique()))

    # 원래 id로 돌려주기 위한 해쉬
    id2profile_for_infernece = dict((i, pid) for (i, pid) in enumerate(raw_data.user.unique()))
    id2show_for_inference = dict((i,sid) for i,sid in enumerate(raw_data['item'].unique()))

    ## ready data for inference ##
    raw_numerize = numerize(raw_data, profile2id_for_infernece, show2id_for_inference)

    # sparse matrix 만들기
    n_users = raw_numerize['uid'].max() + 1
    n_items = raw_numerize['sid'].nunique()

    rows, cols = raw_numerize['uid'], raw_numerize['sid']
    raw_numerize_data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, n_items))

    # 모델에 넣어주고, 아웃풋 리턴받기
    raw_numerize_data_tensor = naive_sparse2tensor(raw_numerize_data).to(args.device)
    recon_batch, mu, logvar = model(raw_numerize_data_tensor)

    # 각 아이템들에 대한 확률값, 후에 sigmoid 취해주는 듯 (근데 그냥 큰 값이면 확률 높게 예측하니까 큰 순서대로 10개뽑음)
    recon_numpy = recon_batch.cpu().detach().numpy()
    recon_numpy[raw_numerize_data_tensor.cpu().detach().numpy().nonzero()] = -np.inf

    # 유저별 10개 추천
    batch_users = recon_numpy.shape[0]

    idx = bn.argpartition(-recon_numpy, 10, axis=1) # 10개 추천
    recon_numpy_binary = np.zeros_like(recon_numpy, dtype=bool)
    recon_numpy_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :10]] = True
    
    # 유저별로 10개씩 담아보기
    result = []
    for id in range(batch_users):
        user_top10 = idx[id][:10]
        for each in user_top10:
            result.append(id2show_for_inference[each])

    users = raw_data['user'].unique().repeat(10)
    inference_df = pd.DataFrame(zip(users,result), columns=['user','item'])
    inference_df.to_csv(f'./output/{args.time_info}_{args.model}_submission.csv',index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')


    parser.add_argument('--data', type=str, default='/opt/ml/level2_movierecommendation_recsys-level2-recsys-11/data/train/',
                        help='Movielens dataset location')
    
    parser.add_argument('--model', type=str, default='VAE',
                        help='select model, DAE or VAE')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args([])
    update_count = 0
    main(args)