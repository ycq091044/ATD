from scipy.io import loadmat
import pandas as pd
import pickle
import numpy as np
import os
from multiprocessing import Process


def train_val_test(root_folder, k, N, epoch_sec):
    all_index = ["/HR{:0>5}".format(i+1) for i in range(21837)]
    
    train_index = np.random.choice(all_index, int(len(all_index) * 0.8), replace=False)
    test_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.1), replace=False)
    val_index = list(set(all_index) - set(train_index) - set(test_index))

    sample_package(root_folder, k, N, epoch_sec, 'pretext', train_index)
    sample_package(root_folder, k, N, epoch_sec, 'train', test_index)
    sample_package(root_folder, k, N, epoch_sec, 'test', val_index)


def sample_package(root_folder, k, N, epoch_sec, train_test_val, index):
    for i, j in enumerate(index):
        if i % N == k:
            print ('train', i, j, 'finished')

            raw_filename = root_folder + j + '.mat'
            lab_filename = root_folder + j + '.hea'

            # X load
            X = loadmat(raw_filename)['val']
            y = open(lab_filename, 'r').readlines()[14][:-1].split(' ')[-1]

            # print (X.shape[1])

            for index in range(X.shape[1] // 2500 - 1):
                path = '/srv/local/data/WFDB/processed/{}/{}-{}.pkl'.format(train_test_val, j, index)
                print ('finish /srv/local/data/WFDB/processed/{}{}-{}.pkl'.format(train_test_val, j, index))
                pickle.dump({'X': X[:, 2500*index:2500*index+5000], 'y': y}, open(path, 'wb'))



if __name__ == '__main__':
    if not os.path.exists('/srv/local/data/WFBD/processed/'):
        os.makedirs('/srv/local/data/WFDB/processed/pretext')
        os.makedirs('/srv/local/data/WFDB/processed/train')
        os.makedirs('/srv/local/data/WFDB/processed/test')

    root_folder = '/srv/local/data/WFDB/raw'

    N, epoch_sec = 30, 10
    p_list = []
    for k in range(N):
        process = Process(target=train_val_test, args=(root_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()