# ====================================================
# CFG
# ====================================================
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--seed', default=42, type=int)
    arg('--use_cuda_if_available', default=True, type=bool)
    arg('--wandb', default=True, type=bool)

    arg('--basepath', default='../data/train', type=str, help='data directory')
    arg('--model_save_path', default='./output', type=str, help='model directory')
    arg('--ii_cons_mat_path', default='./output/ii_constraint_mat', type=str)
    arg('--ii_neigh_mat_path', default='./output/ii_neigh_mat_path', type=str)

    arg('--negative_num', default=200, type=int, help='number of negative samples for each positive pair')
    arg('--negative_weight', default=200, type=int)
    arg('--ii_neighbor_num', default=10, type=int, help='K number of similar items')
    arg('--topk', default=10, type=int)
    arg('--train_ratio', default=0.9, type=float)

    arg('--embedding_dim', default=64, type=int)
    arg('--lr', default=1e-3, type=float)
    arg('--max_epoch', default=1000, type=int)
    arg('--drop_rate', default=0.1, type=float)
    arg('--early_stop_epoch', default=50, type=int)
    arg('--batch_size', default=1024, type=int)

    #L = -(w1 + w2*\beta)) * log(sigmoid(e_u e_i)) - \sum_{N-} (w3 + w4*\beta) * log(sigmoid(e_u e_i'))
    arg('--w1', default=1e-9, type=float)
    arg('--w2', default=1.0, type=float)
    arg('--w3', default=1e-9, type=float)
    arg('--w4', default=1.0, type=float)

    arg('--initial_weight', default=1e-4, type=float, help='initial weight of user and item embeddings')

    arg('--GAMMA', default=1e-4, type=float, help='weight of l2 normalization')
    arg('--LAMBDA', default=1e-3, type=float, help='weight of L_I')

    arg('--sampling_sift_pos', default=True, help='whether to sift the pos item when doing negative sampling')

    args = parser.parse_args()

    return args