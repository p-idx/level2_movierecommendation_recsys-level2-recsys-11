import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--seed', default=42, type=int)
    arg('--use_cuda_if_available', default=True, type=bool)

    arg('--datapath', default='../data/train', type=str, help='data directory')
    arg('--output_dir', default='./output', type=str)

    arg('--num_negative', default=50, type=int, help='number of negative samples')
    arg('--topk', default=10, type=int)
    arg('--train_ratio', default=0.9, type=float)

    arg('--embedding_dim', default=64, type=int)
    arg('--mlp_dims', default=[30, 20, 10], type=list)
    arg('--lr', default=1e-3, type=float)
    arg('--num_epochs', default=50, type=int)
    arg('--drop_rate', default=0.1, type=float)
    arg('--early_stopping', default=100, type=int)

    args = parser.parse_args()

    return args