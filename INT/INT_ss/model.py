"""
ResNet on Spectrogram
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import random


# --- SLEEP ---
class AE_SLEEP(nn.Module):

    def __init__(self, n_dim):
        super(AE_SLEEP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            ResBlock(16, 32, 2, True, False),
            ResBlock(32, 64, 2, True, True),
            ResBlock(64, 64, 1, True, True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=3),
            nn.BatchNorm2d(8),
            nn.ELU(True),
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(True),
            nn.ConvTranspose2d(16, 14, 2, stride=2),
            nn.BatchNorm2d(14),
            nn.ELU(True),
            nn.ConvTranspose2d(14, 14, 2, stride=2),
        )

        self.down = nn.Sequential(
            nn.ELU(True),
            nn.Linear(128, n_dim, bias=True),
        )

        self.up = nn.Sequential(
            nn.ELU(True),
            nn.Linear(n_dim, 128, bias=True)
        )
    
    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 // 8,
                center = False,
                onesided = True,
                # normalized = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1)+1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, mid=True):
        y = self.torch_stft(x)
        x = self.encoder(y)
        rep = x.reshape(x.shape[0], -1)
        if mid:
            return rep
        x = rep.reshape(rep.shape[0], 4, 8, 4)
        x = self.decoder(x)[:, :, :y.shape[2], :y.shape[3]]
        return x, y, rep

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(4, stride=4) 
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

class CNNEncoder2D_SLEEP(nn.Module):
    def __init__(self, n_dim):
        super(CNNEncoder2D_SLEEP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(16, 32, 2, True, False)
        self.conv3 = ResBlock(32, 64, 2, True, True)
        self.conv4 = ResBlock(64, 64, 1, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Linear(128, self.n_dim, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 // 8,
                center = False,
                onesided = True,
                normalized = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1)+1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, mid=True, byol=False):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)

        if mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x


# --- MGH ---
class AE_MGH(nn.Module):
    def __init__(self, n_dim):
        super(AE_MGH, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=(3,2), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResBlock_MGH(16, 32, (3, 2), True, False),
            ResBlock_MGH(32, 64, (3, 2), True, False),
            ResBlock_MGH(64, 64, 2, True, True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, (6, 5), stride=(6, 5)), 
            nn.BatchNorm2d(8),
            nn.ELU(True),
            nn.ConvTranspose2d(8, 12, (3, 3), stride=(3, 3)), 
            nn.BatchNorm2d(12),
            nn.ELU(True),
            nn.ConvTranspose2d(12, 12, (2, 1), stride=(1, 1), padding=(16,1)), 
        )

        self.down = nn.Sequential(
            nn.ELU(True),
            nn.Linear(192, n_dim, bias=True),
        )

        self.up = nn.Sequential(
            nn.ELU(True),
            nn.Linear(n_dim, 192, bias=True)
        )
    
    def torch_stft(self, X_train):
        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 512,
                hop_length = 512 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, mid=True):
        y = self.torch_stft(x)
        x = self.encoder(y)
        rep = self.down(x.reshape(x.shape[0], -1))
        if mid:
            return rep
        x = self.up(rep).reshape(rep.shape[0], 4, 16, 3)
        x = self.decoder(x)
        return x, y, rep
        
class ResBlock_MGH(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock_MGH, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(4, stride=4) 
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

class CNNEncoder2D_MGH(nn.Module):
    def __init__(self, n_dim):
        super(CNNEncoder2D_MGH, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=(3,2), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = ResBlock_MGH(16, 32, (3, 2), True, False)
        self.conv3 = ResBlock_MGH(32, 64, (3, 2), True, False)
        self.conv4 = ResBlock_MGH(64, 64, 2, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(192, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.Linear(self.n_dim, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 512,
                hop_length = 512 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, siame=False, mid=True, byol=False, sup=False):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)

        if sup:
            return self.sup(x)
        elif siame:
            return x, self.fc(x)
        elif mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x
            

# --- WFDB --- 
class AE_WFDB(nn.Module):

    def __init__(self, n_dim):
        super(AE_WFDB, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            ResBlock_WFDB(32, 64, 3, True, False),
            ResBlock_WFDB(64, 64, 3, True, True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=3),
            nn.BatchNorm2d(16),
            nn.ELU(True),
            nn.ConvTranspose2d(16, 24, 3, stride=3),
            nn.BatchNorm2d(24),
            nn.ELU(True),
            nn.ConvTranspose2d(24, 24, 2, stride=2),
        )

        self.down = nn.Sequential(
            nn.ELU(True),
            nn.Linear(64*3*2, n_dim, bias=True),
        )

        self.up = nn.Sequential(
            nn.ELU(True),
            nn.Linear(n_dim, 64*3*2, bias=True)
        )
    
    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, mid=True):
        y = self.torch_stft(x)
        x = self.encoder(y)
        rep = self.down(x.reshape(x.shape[0], -1))
        if mid:
            return rep
        x = self.up(rep).reshape(rep.shape[0], 8, 8, 6)
        # print (self.decoder(x).shape)
        x = self.decoder(x)[:, :, :y.shape[2], :y.shape[3]]
        return x, y, rep

class ResBlock_WFDB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock_WFDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(4, stride=4) 
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

class CNNEncoder2D_WFDB(nn.Module):
    def __init__(self, n_dim):
        super(CNNEncoder2D_WFDB, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock_WFDB(32, 64, 3, True, False)
        self.conv3 = ResBlock_WFDB(64, 64, 3, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.ELU(),
            nn.Linear(384, self.n_dim, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, mid=True, byol=False):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)

        if mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x


# --- HAR ---
class AE_HAR(nn.Module):

    def __init__(self, n_dim):
        super(AE_HAR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(18),
            nn.ELU(inplace=True),
            ResBlock(18, 32, 2, True, False),
            ResBlock(32, 32, 2, True, True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=3),
            nn.BatchNorm2d(16),
            nn.ELU(True),
            nn.ConvTranspose2d(16, 18, 3, stride=3),
        )

        self.down = nn.Sequential(
            nn.ELU(True),
            nn.Linear(128, n_dim, bias=True),
        )

        self.up = nn.Sequential(
            nn.ELU(True),
            nn.Linear(n_dim, 128, bias=True)
        )
    
    def torch_stft(self, X_train):
        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 64,
                hop_length = 2,
                center = False,
                onesided = True,
                # normalized = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1)+1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, mid=True):
        y = self.torch_stft(x)
        x = self.encoder(y)
        rep = self.down(x.reshape(x.shape[0], -1))
        if mid:
            return rep
        x = self.up(rep).reshape(rep.shape[0], 8, 4, 4)
        x = self.decoder(x)[:, :, :y.shape[2], :y.shape[3]]
        return x, y, rep

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(4, stride=4) 
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

class CNNEncoder2D_HAR(nn.Module):
    def __init__(self, n_dim):
        super(CNNEncoder2D_HAR, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(18, 18, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(18),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(18, 32, 2, True, False)
        self.conv3 = ResBlock(32, 32, 2, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.ELU(),
            nn.Linear(128, self.n_dim, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 64,
                hop_length = 2,
                center = False,
                onesided = True,
                # normalized = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, mid=True, byol=False):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        if mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x
