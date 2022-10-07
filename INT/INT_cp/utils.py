import torch
import numpy as np
from scipy.signal import spectrogram
import pickle
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram
from transforms3d.axangles import axangle2mat  # for rotation

def denoise_channel(ts, bandpass, signal_freq, bound):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    low_length = np.random.randint(len_ts//20, len_ts//2)

    # high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    # low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(low_length)-1)
        x_old = np.linspace(0, 1, num=low_length, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    # both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(low_length)-1)
        x_old = np.linspace(0, 1, num=low_length, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts
        
    return out_ts

class SLEEPLoader(torch.utils.data.Dataset):

    def __init__(self, list_IDs, dir, SS=0):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 30)
        self.bandpass2 = (10, 49)
        self.n_channels = 7
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)

    def jittering(self, x, ratio, deg):
        """
        Add noise to multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=deg, bound=self.bound)
        return x
    
    def bandpas_filtering(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            else:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
        return x
    
    def augment(self, x):
        t = np.random.rand()
        if t > 0.5:
            x = self.jittering(x, ratio=0.5, deg=0.05)
        else:
            x = self.bandpas_filtering(x, ratio=0.5)
        return x
    
    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], sample['y']
        
        # raw y in ['W', 'R', 0, 1, 2, 3, 5]
        # we map: 'W'->0, '1'->1, '2'->2, '3','4'->3, 'R'->4, others->0
        if y == 'W':
            y = 0
        elif y == 'R':
            y = 4
        elif y in ['1', '2', '3']:
            y = int(y)
        elif y == '4':
            y = 3
        else:
            y = 0
        
        y = torch.LongTensor([y])

        if self.SS == 0:
            return torch.FloatTensor(X)
        elif self.SS == 1:
            return torch.FloatTensor(self.augment(X.copy()))
        elif self.SS == 2:
            return torch.FloatTensor(X.copy()), torch.FloatTensor(self.augment(X.copy()))
        elif self.SS == 3:
            return torch.FloatTensor(X), y
        elif self.SS == 4:
            return torch.FloatTensor(self.augment(X.copy())), y

class MGHLoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=0, aug=None):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS
        self.aug = aug
        self.label_list = [1, 2, 3, 4, 5]
        self.bandpass1 = (0.001, 30)
        self.bandpass2 = (10, 50)
        self.n_length = 200 * 30
        self.n_channels = 6
        self.n_classes = 5
        self.signal_freq = 200
        self.bound = 50

    def __len__(self):
        return len(self.list_IDs)

    def augment(self, x, aug=None):
        t = np.random.rand()
        if aug is None:
            if t > 0.5:
                x = self.jittering(x, ratio=0.5)
            else:
                x = self.bandpas_filtering(x, ratio=0.5)
        else:
            aug_list = aug.split(',')
            aug_method = []
            if 'A' in aug_list:
                aug_method.append(self.jittering)
            if 'B' in aug_list:
                aug_method.append(self.bandpas_filtering)

            x = np.random.choice(aug_method, 1)[0](x)
        return x

    def jittering(self, x, ratio=0.5):
        """
        Add noise to multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            mode = np.random.choice(['high', 'low', 'both'])
            x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.01, bound=self.bound)
        return x
    
    def bandpas_filtering(self, x, ratio=0.5):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.5:
                x[i, :] = denoise_channel(x[i, :].copy(), self.bandpass1, self.signal_freq, bound=self.bound)
            else:
                x[i, :] = denoise_channel(x[i, :].copy(), self.bandpass2, self.signal_freq, bound=self.bound)
        return x

    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'][:, :], sample['y']

        # raw y in [1, 2, 3, 4, 5]
        y = self.label_list[y]

        y = torch.LongTensor([y])

        if self.SS == 0:
            return torch.FloatTensor(X)
        elif self.SS == 1:
            return torch.FloatTensor(self.augment(X.copy()))
        elif self.SS == 2:
            return torch.FloatTensor(X.copy()), torch.FloatTensor(self.augment(X.copy(), self.aug))
        elif self.SS == 3:
            return torch.FloatTensor(X), y
        elif self.SS == 4:
            return torch.FloatTensor(self.augment(X.copy())), y

class WFDBLoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=0):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_map = ['Male', 'Female']
        self.bandpass1 = (0.001, 30)
        self.bandpass2 = (10, 50)
        self.n_length = 500 * 10
        self.n_channels = 12
        self.n_classes = 2
        self.signal_freq = 500
        self.bound = 100

    def __len__(self):
        return len(self.list_IDs)

    def augment(self, x):
        t = np.random.rand()
        if t > 0.5:
            x = self.jittering(x, ratio=0.5)
        else:
            x = self.bandpas_filtering(x, ratio=0.5)
        return x

    def jittering(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            mode = np.random.choice(['high', 'low', 'both'])
            x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.001, bound=self.bound)
        return x
    
    def bandpas_filtering(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            else:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
        return x

    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], self.label_map.index(sample['y'])

        if self.SS == 0:
            return torch.FloatTensor(X)
        elif self.SS == 1:
            return torch.FloatTensor(self.augment(X.copy()))
        elif self.SS == 2:
            return torch.FloatTensor(X.copy()), torch.FloatTensor(self.augment(X.copy()))
        elif self.SS == 3:
            return torch.FloatTensor(X), y
        elif self.SS == 4:
            return torch.FloatTensor(self.augment(X.copy())), y

def threeD_pos_rotation(X):
    """
    https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
    """
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi / 4, high=np.pi / 4)
    return np.matmul(X , axangle2mat(axis,angle))

class HARLoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=0):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS
        self.n_length = None
        self.label_map = ['1', '2', '3', '4', '5', '6']
        self.bandpass1 = (1e-6, 20)
        self.bandpass2 = (5, 24.5)
        self.n_length = 128
        self.n_channels = 9
        self.n_classes = 6
        self.signal_freq = 50

    def __len__(self):
        return len(self.list_IDs)
        
    def cellphone_rotation(self, x):
        for i in range(3):
            x[i*3:i*3+3, :] = threeD_pos_rotation(x[i*3:i*3+3, :].T).T
        return x

    def augment(self, x):
        t = np.random.rand()
        if t > 0.66:
            x = self.jittering(x, ratio=0.5)
        elif t > 0.33:
            x = self.bandpas_filtering(x, ratio=0.5)
        else:
            x = self.cellphone_rotation(x)
        return x

    def jittering(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.002, bound = 0)
        return x
    
    def bandpas_filtering(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.5:
                x[i, :] = denoise_channel(x[i, :].copy(), self.bandpass1, self.signal_freq, bound = 0)
            else:
                x[i, :] = denoise_channel(x[i, :].copy(), self.bandpass2, self.signal_freq, bound = 0)
        return x

    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], self.label_map.index(sample['y'])

        if self.SS == 0:
            return torch.FloatTensor(X)
        elif self.SS == 1:
            return torch.FloatTensor(self.augment(X.copy()))
        elif self.SS == 2:
            return torch.FloatTensor(X.copy()), torch.FloatTensor(self.augment(X.copy()))
        elif self.SS == 3:
            return torch.FloatTensor(X), y
        elif self.SS == 4:
            return torch.FloatTensor(self.augment(X.copy())), y

 



