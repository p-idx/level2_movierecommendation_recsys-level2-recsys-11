import time
import torch

from args import parse_args
from dataset import preprocess, data_loader
from trainer import train
from utils import set_seed, check_path

def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    set_seed(args.seed)
    check_path(args.output_dir)

    # Preprocess
    data, field_dims, users, items = preprocess(args)

    # Dataloader
    train_loader, valid_loader = data_loader(args, data, field_dims)

    # Train
    train(args, device, field_dims, train_loader, valid_loader)
    


if __name__ == "__main__":
    main()