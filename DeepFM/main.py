import time
import torch

from args import parse_args
from dataset import preprocess, data_loader
from trainer import train
from utils import set_seed, check_path
import os


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    set_seed(args.seed)
    check_path(args.output_dir)

    # Preprocess
    data, field_dims, idx_dict = preprocess()

    # Dataloader
    train_loader, valid_loader = data_loader(args, data)

    # Train
    train(args, device, field_dims, train_loader, valid_loader)
    


if __name__ == "__main__":
    main()