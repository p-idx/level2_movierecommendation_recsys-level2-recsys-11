# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True


    ##### MODEL

    embedding_dim=64

    ii_neighbor_num=10
    model_save_path="./weight/best_model.pt"
    max_epoch=1000
    # enable_tensorboard=yes
    initial_weight=1e-4


    ################
    ##### Training

    basepath="./data/train"

    #need to specify the avaliable gpu index. If gpu is not avaliable, we will use cpu.
    gpu=0
    num_workers=1

    lr=1e-3
    batch_size=512
    early_stop_epoch=25

    #L = -(w1 + w2*\beta)) * log(sigmoid(e_u e_i)) - \sum_{N-} (w3 + w4*\beta) * log(sigmoid(e_u e_i'))
    w1=1e-9
    w2=1
    w4=1e-9
    w3=1

    negative_num= 200
    negative_weight= 200

    #weight of l2 normalization
    GAMMA=1e-4
    #weight of L_I
    LAMBDA=1e-3

    #whether to sift the pos item when doing negative sampling
    sampling_sift_pos=True

    #################
    #### Testing

    #can be customized to your gpu size
    test_batch_size=2048
    topk=20
    test_file_path="./data"