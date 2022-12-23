import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os

from sklearn.metrics import accuracy_score, roc_auc_score
from model import DeepFM

def test(device, model, valid_loader):
    with torch.no_grad():
        model.eval()
        total_pred = []; total_ans = []

        for fields, target in valid_loader:
            fields = fields.to(device)
            result = model(fields)

            result = result.detach().cpu().tolist()
            target = target.tolist()

            total_pred.extend(result)
            total_ans.extend(target)
        
    total_pred = np.array(total_pred)
    total_ans = np.array(total_ans)

    auc = roc_auc_score(total_ans, total_pred)

    rounded_pred = np.rint(total_pred)
    acc = accuracy_score(total_ans, rounded_pred)

    return auc, acc

def train(args, device, field_dims, train_loader, valid_loader):
    input_dims = field_dims
    model = DeepFM(input_dims, args.embedding_dim, args.mlp_dims).to(device)
    bce_loss = nn.BCELoss() # Binary Cross Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_epoch, best_auc, best_acc, early_stopping = 0, 0, 0, 0
    print('TRAINING...')

    for epoch in tqdm(range(args.num_epochs)):
        for fields, target in train_loader:  # train
            fields, target = fields.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(fields)
            loss = bce_loss(output, target.float())
            loss.backward()
            optimizer.step()

        AUC, ACC = test(device, model, valid_loader)   # validation

        print(f"EPOCH({epoch})  TRAIN LOSS: {loss:.3f}  VALID AUC: {AUC:.3f}    VALID ACC: {ACC:.3f}")

        if AUC > best_auc:
            best_epoch, best_auc, best_acc = epoch, AUC, ACC
            early_stopping = 0
            torch.save(model.state_dict(), model_dir)
        
        else:
            early_stopping += 1
            if early_stopping == args.early_stopping:
                print('##########################')
                print(f'Early stopping triggered at epoch {epoch}')
                print(f'BEST AUC: {best_auc}, ACC: {best_acc}, BEST EPOCH: {best_epoch}')

                model_dir = os.path.join(args.output_dir, "model.pt")
                break
                
    print('TRAINING DONE!')