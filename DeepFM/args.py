import argparse

def parse_args():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--seed', default=42, type=int)
    arg('--use_cuda_if_available', default=True, type=bool)

    arg('--data_path', default='../data/train', type=str, help='data directory')
    arg('--output_dir', default='./output', type=str)

    arg('--ratio_negative', default=0.4, type=int, help='number of negative samples')
    arg('--ratio_negative_long', default=0.2, type=int, help='number of negative samples')
    arg('--negative_threshold', default=500, type=int, help='threshold of negative samples')
    arg('--topk', default=10, type=int)
    arg('--train_ratio', default=0.9, type=float)
    arg('--train', default='False', type=str2bool)


    arg('--embedding_dim', default=64, type=int)
    arg('--mlp_dims', default=[30, 20, 10], type=list)
    arg('--lr', default=1e-3, type=float)
    arg('--num_epochs', default=14, type=int)
    arg('--drop_rate', default=0.1, type=float)
    arg('--early_stopping', default=100, type=int)

    args = parser.parse_args()

    return args