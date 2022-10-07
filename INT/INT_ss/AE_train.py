import torch
from utils import SLEEPLoader, MGHLoader, WFDBLoader, HARLoader
import numpy as np
import torch.nn as nn
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from model import AE_SLEEP, AE_HAR, AE_MGH, AE_WFDB
from loss import SimCLR, BYOL
from tqdm import tqdm
from collections import Counter
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help="number of epochs")
parser.add_argument('--lr', type=float, default=0.5e-3, help="learning rate")
parser.add_argument('--n_dim', type=int, default=32, help="hidden units")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
parser.add_argument('--m', type=float, default=0.995, help="moving coefficient")
parser.add_argument('--model', type=str, default='Ours', help="which model")
parser.add_argument('--T', type=float, default=1.0,  help="T")
parser.add_argument('--sigma', type=float, default=2.0,  help="sigma")
parser.add_argument('--delta', type=float, default=0.2,  help="delta")
parser.add_argument('--cuda', type=str, default='1', help="which gpu")
parser.add_argument('--dataset', type=str, default='SLEEP', help="SLEEP, MGH, HAR, WFDB")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

# linear evaluation
def task(X_train, X_test, y_train, y_test):
            
    cls = LR(solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(X_train, y_train)
    pred = cls.predict(X_test)
    
    res = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    return res, cm

# unsupervised learning
def Pretext(q_encoder, optimizer, Epoch, pretext_loader, train_loader, test_loader, criterion):

    q_encoder.train()
    step = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)
    semi_result, all_loss, acc_score = [], [], []
    for epoch in range(Epoch):
        for index, (T_aug1, T_aug2) in enumerate(tqdm(pretext_loader)):
            if args.model == 'AE':
                T_aug = T_aug1.to(device)
                rec_aug, stft_aug, rep = q_encoder(T_aug, mid=False)
                loss = torch.norm(rec_aug - stft_aug) ** 2 / torch.norm(stft_aug) ** 2
            elif args.model == 'AE_SS':
                T_aug1 = T_aug1.to(device)
                T_aug2 = T_aug2.to(device)
                rec_aug1, stft_aug1, rep1 = q_encoder(T_aug1, mid=False)
                rec_aug2, stft_aug2, rep2 = q_encoder(T_aug2, mid=False)
                loss = torch.norm(rec_aug1 - stft_aug1) ** 2 / torch.norm(stft_aug1) ** 2 + \
                        torch.norm(rec_aug2 - stft_aug2) ** 2 / torch.norm(stft_aug2) ** 2 + 1/2 * criterion(rep1, rep2)
            # loss back
            all_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # only update encoder_q

        if epoch % 10 == 9:
            semi_result.append(evaluate(q_encoder, train_loader, test_loader))
            print (semi_result)

    return q_encoder, semi_result

def evaluate(q_encoder, train_loader, test_loader):
    # freeze
    q_encoder.eval()

    emb_val, gt_val = [], []
    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val, mid=True).cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)

    emb_test = []
    gt_test = []
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test, mid=True).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())
    emb_test, gt_test= np.array(emb_test), np.array(gt_test)           

    res, cm = task(emb_val, emb_test, gt_val, gt_test)
    q_encoder.train()
    return res

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if args.dataset == 'SLEEP':
        # dataset
        pretext_dir = '/srv/local/data/SLEEPEDF/processed/train/'
        train_dir = '/srv/local/data/SLEEPEDF/processed/val/'
        test_dir = '/srv/local/data/SLEEPEDF/processed/test/'

        pretext_index = np.random.choice(os.listdir(pretext_dir), 50000, replace=False)
        train_index = np.random.choice(os.listdir(train_dir), 5000, replace=False)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(SLEEPLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(SLEEPLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)

        pretext_loader = torch.utils.data.DataLoader(SLEEPLoader(pretext_index, pretext_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # initial parameters
        q_encoder = AE_SLEEP(args.n_dim)
    
    elif args.dataset == 'MGH':
        # dataset
        pretext_dir = '/srv/local/data/MGH/train/'
        train_dir = '/srv/local/data/MGH/val/'
        test_dir = '/srv/local/data/MGH/test/'

        pretext_index = np.random.choice(os.listdir(pretext_dir), 50000, replace=False)
        train_index = np.random.choice(os.listdir(train_dir), 5000, replace=False)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(MGHLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(MGHLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)

        pretext_loader = torch.utils.data.DataLoader(MGHLoader(pretext_index, pretext_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # initial parameters
        q_encoder = AE_MGH(args.n_dim)
    
    elif args.dataset == 'HAR':
        # dataset
        pretext_dir = '/srv/local/data/HAR/processed/pretext/'
        train_dir = '/srv/local/data/HAR/processed/train/'
        test_dir = '/srv/local/data/HAR/processed/test/'

        pretext_index = os.listdir(pretext_dir)
        train_index = os.listdir(train_dir)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(HARLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(HARLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)

        pretext_loader = torch.utils.data.DataLoader(HARLoader(pretext_index, pretext_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # initial parameters
        q_encoder = AE_HAR(args.n_dim)

    elif args.dataset == 'WFDB':
        # dataset
        pretext_dir = '/srv/local/data/WFDB/processed/pretext/'
        train_dir = '/srv/local/data/WFDB/processed/train/'
        test_dir = '/srv/local/data/WFDB/processed/test/'

        pretext_index = os.listdir(pretext_dir)
        train_index = os.listdir(train_dir)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(WFDBLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(WFDBLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        pretext_loader = torch.utils.data.DataLoader(WFDBLoader(pretext_index, pretext_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # define model
        q_encoder = AE_WFDB(args.n_dim)

    q_encoder.to(device)

    print (sum(p.numel() for p in q_encoder.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(q_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = SimCLR(device).to(device)
    # run
    q_encoder, semi_result = Pretext(q_encoder, optimizer, args.epochs, pretext_loader, train_loader, test_loader, criterion)