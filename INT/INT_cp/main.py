import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, accuracy_score
from model import SALS, ATD, GR_SALS
from utils import SLEEPLoader, MGHLoader, HARLoader, WFDBLoader
import time
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help="number of epochs")
parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
parser.add_argument('--R', type=int, default=32, help="hidden units")
parser.add_argument('--epsilon', type=float, default=5, help="beta in the paper")
parser.add_argument('--reg', type=float, default=1, help="alpha in the paper")
parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
parser.add_argument('--model', type=str, default='ATD', help="ATD or SALS")
parser.add_argument('--cuda', type=str, default='1', help="which gpu?")
parser.add_argument('--dataset', type=str, default='SLEEP', help="SLEEP, MGH, HAR, or WFDB")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

# logistic regression
def task(X_train, X_test, y_train, y_test):
            
    cls = LR(solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(X_train, y_train)
    # cls = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(32,)).fit(X_train, y_train)
    pred_lr_train = cls.predict(X_train)
    pred_lr_test = cls.predict(X_test)

    res_train = accuracy_score(y_train, pred_lr_train)
    res_test = accuracy_score(y_test, pred_lr_test)
    
    return res_train, res_test

# the feature extractor
def inference(T, param, device):
    """
    param: {A, B, C}
    """
    A, B, C = param
    X = torch.solve(torch.einsum('ijkl,jr,kr,lr->ri',T,A,B,C), (A.T@A)*(B.T@B)*(C.T@C) + torch.eye(C.shape[1]).to(device)*args.reg)[0].T
    return X
    
# linear evaluation
def evaluate(param, train_loader, test_loader, device):

    emb_train, gt_train = [], []
    for X_train, y_train in train_loader:
        X_train = torch_stft(X_train.squeeze(dim=1).to(device))
        emb_train.extend(inference(X_train, param, device).cpu().tolist())
        gt_train.extend(y_train.numpy().flatten())
    emb_train, gt_train = np.array(emb_train), np.array(gt_train)

    emb_val, gt_val = [], []
    for X_val, y_val in test_loader:
        X_val = torch_stft(X_val.squeeze(dim=1).to(device))
        emb_val.extend(inference(X_val, param, device).cpu().tolist())
        gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val) 

    res_train, res_test = task(emb_train, emb_val, gt_train, gt_val)
    return res_train, res_test

# the STFT transformation:
# raw input sample -> fourth-order tensor sample
def torch_stft(X_train):
    """
    input: channel x timestamp
    output: channel x frequency x timewindow
    """
    signal = []
    for s in range(X_train.shape[1]):

        if args.dataset == 'SLEEP':
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 // 8,
                center = False,
                onesided = True,
                # normalized = True,
                return_complex=False)

        elif args.dataset == 'MGH':
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 512,
                hop_length = 512 // 4,
                center = False,
                onesided = True,
                # normalized = True,
                return_complex=False)

        elif args.dataset == 'HAR':
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 64,
                hop_length = 2,
                center = False,
                onesided = True,
                normalized = True,
                return_complex=False)

        elif args.dataset == 'WFDB':
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 // 4,
                center = False,
                onesided = True,
                # normalized = True,
                return_complex=False)

        signal.append(spectral)
    
    # after STFT, we stack the real and imagenary part as different channels, so channel doubled.
    signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
    signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

    # finally, we do log-transformation for the fourth-order tensor
    return torch.cat([torch.log(torch.abs(signal1) + 1e-3), torch.log(torch.abs(signal2) + 1e-3)], dim=1)

def train_aug(A, B, C, unlabeled_loader_aug, device):
    rec_list = []
    for index, (T_batch, T_batch_aug) in enumerate(tqdm(unlabeled_loader_aug)):
        T_batch = torch_stft(T_batch.squeeze(dim=1).to(device))
        T_batch_aug = torch_stft(T_batch_aug.squeeze(dim=1).to(device))

        [A, B, C], rec = ATD(A, B, C, T_batch, T_batch_aug, args.lr, device, args.reg, args.epsilon)
        rec_list.append(rec.item())

    return [A, B, C], rec_list


def train_CPD(A, B, C, unlabeled_loader, device):
    rec_list = []
    for index, T_batch in enumerate(tqdm(unlabeled_loader)):
        T_batch = torch_stft(T_batch.squeeze(dim=1).to(device))

        [A, B, C], rec = SALS(A, B, C, T_batch, args.lr, device, args.reg)
        rec_list.append(rec.item())
    return [A, B, C], rec_list

def train_CPD_graph_regularizer(A, B, C, unlabeled_loader, device):
    rec_list = []
    for index, T_batch in enumerate(tqdm(unlabeled_loader)):
        T_batch = torch_stft(T_batch.squeeze(dim=1).to(device))

        [A, B, C], rec = GR_SALS(A, B, C, T_batch, args.lr, device, args.reg)
        rec_list.append(rec.item())
    return [A, B, C], rec_list

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if args.dataset == 'SLEEP':
        # data path
        unlabeled_dir = '/srv/local/data/SLEEPEDF/processed/train/'
        train_dir = '/srv/local/data/SLEEPEDF/processed/val/'
        test_dir = '/srv/local/data/SLEEPEDF/processed/test/'

        unlabeled_index = np.random.choice(os.listdir(unlabeled_dir), 50000, replace=False)
        train_index = np.random.choice(os.listdir(train_dir), 5000, replace=False)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(SLEEPLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(SLEEPLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)

        unlabeled_loader_CPD = torch.utils.data.DataLoader(SLEEPLoader(unlabeled_index, unlabeled_dir, 0), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_Random = torch.utils.data.DataLoader(SLEEPLoader(unlabeled_index, unlabeled_dir, 1), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_aug = torch.utils.data.DataLoader(SLEEPLoader(unlabeled_index, unlabeled_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # initial parameters
        A = torch.FloatTensor(torch.randn(14, args.R)).to(device)
        B = torch.FloatTensor(torch.randn(129, args.R)).to(device)
        C = torch.FloatTensor(torch.randn(86, args.R)).to(device)
    
    elif args.dataset == 'MGH':
        # data path
        unlabeled_dir = '/srv/local/data/MGH/train/'
        train_dir = '/srv/local/data/MGH/val/'
        test_dir = '/srv/local/data/MGH/test/'

        unlabeled_index = np.random.choice(os.listdir(unlabeled_dir), 50000, replace=False)
        train_index = np.random.choice(os.listdir(train_dir), 5000, replace=False)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(MGHLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(MGHLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)

        unlabeled_loader_CPD = torch.utils.data.DataLoader(MGHLoader(unlabeled_index, unlabeled_dir, 0), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_Random = torch.utils.data.DataLoader(MGHLoader(unlabeled_index, unlabeled_dir, 1), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_aug = torch.utils.data.DataLoader(MGHLoader(unlabeled_index, unlabeled_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # initial parameters
        A = torch.FloatTensor(torch.randn(12, args.R)).to(device)
        B = torch.FloatTensor(torch.randn(257, args.R)).to(device)
        C = torch.FloatTensor(torch.randn(43, args.R)).to(device)
    
    elif args.dataset == 'HAR':
        # dataset
        unlabeled_dir = '/srv/local/data/HAR/processed/pretext/'
        train_dir = '/srv/local/data/HAR/processed/train/'
        test_dir = '/srv/local/data/HAR/processed/test/'

        unlabeled_index = os.listdir(unlabeled_dir)
        train_index = os.listdir(train_dir)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(HARLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(HARLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)

        unlabeled_loader_CPD = torch.utils.data.DataLoader(HARLoader(unlabeled_index, unlabeled_dir, 0), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_Random = torch.utils.data.DataLoader(HARLoader(unlabeled_index, unlabeled_dir, 1), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_aug = torch.utils.data.DataLoader(HARLoader(unlabeled_index, unlabeled_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # initial parameters
        A = torch.FloatTensor(torch.randn(18, args.R)).to(device)
        B = torch.FloatTensor(torch.randn(33, args.R)).to(device)
        C = torch.FloatTensor(torch.randn(33, args.R)).to(device)

    elif args.dataset == 'WFDB':
        # dataset
        unlabeled_dir = '/srv/local/data/WFDB/processed/pretext/'
        train_dir = '/srv/local/data/WFDB/processed/train/'
        test_dir = '/srv/local/data/WFDB/processed/test/'

        unlabeled_index = os.listdir(unlabeled_dir)
        train_index = os.listdir(train_dir)
        test_index = os.listdir(test_dir)

        train_loader = torch.utils.data.DataLoader(WFDBLoader(train_index, train_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)
        test_loader = torch.utils.data.DataLoader(WFDBLoader(test_index, test_dir, 3), 
                        batch_size=args.batch_size, shuffle=False, num_workers=5)

        unlabeled_loader_CPD = torch.utils.data.DataLoader(WFDBLoader(unlabeled_index, unlabeled_dir, 0), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_Random = torch.utils.data.DataLoader(WFDBLoader(unlabeled_index, unlabeled_dir, 1), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)
        unlabeled_loader_aug = torch.utils.data.DataLoader(WFDBLoader(unlabeled_index, unlabeled_dir, 2), 
                        batch_size=args.batch_size, shuffle=True, num_workers=5)

        # initial parameters
        A = torch.FloatTensor(torch.randn(24, args.R)).to(device)
        B = torch.FloatTensor(torch.randn(129, args.R)).to(device)
        C = torch.FloatTensor(torch.randn(75, args.R)).to(device)

    early_stop_check = []

    for i in range(args.epochs):

        if args.model in ['ATD']:
            [A, B, C], rec_list = train_aug(A, B, C, unlabeled_loader_aug, device)
        elif args.model in ['SALS']:
            [A, B, C], rec_list = train_CPD(A, B, C, unlabeled_loader_CPD, device)
        elif args.model in ['GR_SALS']:
            [A, B, C], rec_list = train_CPD_graph_regularizer(A, B, C, unlabeled_loader_CPD, device)

        early_stop_check.append(np.mean(rec_list))
        if (len(early_stop_check) > 3) and ((max(early_stop_check[-3:]) - min(early_stop_check[-3:])) / max(early_stop_check[-3:]) < 1e-3):
            print ('loss does not change, early stop!')
            break

    res_train, res_test = evaluate([A, B, C], train_loader, test_loader, device)

    print ('-------- epoch {} -------'.format(i+1))
    print ('rec: {} LR train: {} LR test: {}'.format(np.mean(rec_list), res_train, res_test))

