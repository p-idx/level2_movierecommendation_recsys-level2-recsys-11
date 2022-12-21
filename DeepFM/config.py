# ====================================================
# CFG
# ====================================================

class CFG:
    use_cuda_if_available = True

    ##### DATASET
    datapath = './data'
    
    num_negative = 50

    ##### MODEL

    embedding_dim = 64
    mlp_dims = [30, 20, 10]
    drop_rate = 0.1 

    ##### TRAINING

    lr = 0.01