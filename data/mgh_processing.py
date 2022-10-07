import sys
import mne
import numpy as np
import os
import pandas as pd
from multiprocessing import Process
import pickle
import scipy.io
import h5py

def get_channel_data(path):
    outpath = '../data/processed/signal/' + \
            path.split('/')[-1].split('.')[0] + '_signal.npy'
    if os.path.exists(outpath): return

    data = mne.io.read_raw_edf(path)
    # get_stage(stage_path)
    raw_data = data.get_data()
    np.save(open(outpath, 'wb'), raw_data)

    """ visualize 14 channels
    plt.figure(figsize=(20,10))
    for i in range(raw_data.shape[0]):
        plt.subplot(raw_data.shape[0], 1, i + 1)
        plt.plot(raw_data[i, :100000])
    plt.show()
    """

def channel_process(signal_path, k, l):
    for i, j in enumerate(os.listdir(signal_path)):
        if i % l == k:
            print (i, j, 'finished')
            get_channel_data(signal_path + j)

def get_stage(path):
    with open(path, 'r') as infile:
        text = infile.read()
        root = ET.fromstring(text)
        stages = [i.text for i in root.find('SleepStages').findall('SleepStage')]

    outpath = '../data/processed/label/' + \
            path.split('/')[-1].split('.')[0][:-10] + '_stages.npy'
    np.save(open(outpath, 'wb'), stages)

def train_val_test(root_folder, mutual_name, k, N, epoch_sec):
    all_index = mutual_name

    train_index = np.random.choice(all_index, int(len(all_index) * 0.9), replace=False)
    test_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.05), replace=False)
    val_index = list(set(all_index) - set(train_index) - set(test_index))

    sample_package(root_folder, k, N, epoch_sec, 'pretext', train_index)
    sample_package(root_folder, k, N, epoch_sec, 'train', test_index)
    sample_package(root_folder, k, N, epoch_sec, 'test', val_index)

def sample_package(root_folder, k, N, epoch_sec, train_test_val, index):
    for i, j in enumerate(index):
        if i % N == k:
            print ('train', i, j, 'finished')

            # X load
            signal = scipy.io.loadmat(root_folder + 'Signal_' + j)['s'][:6, :]
            # y load
            with h5py.File(root_folder + 'Labels_' + j, 'r') as infile:
                stages = infile['stage'][()].flatten()

            for slice_index in range(signal.shape[1] // (200 * 30)):
                stage = np.unique(stages[slice_index * (200 * 30): slice_index * (200 * 30) + 200 * epoch_sec])
                if (len(stage) == 1) and (stage[0] in [1, 2, 3, 4, 5]):
                    path = root_folder + 'processed/{}/'.format(train_test_val) + 'mgh-' + str(j)[:-4] + '-' + str(slice_index) + '.pkl'
                    if not os.path.exists(path):
                        pickle.dump({'X': signal[:, slice_index * (200 * 30): slice_index * (200 * 30) + 200 * epoch_sec], \
                            'y': int(stage[0])}, open(path, 'wb'))
            os.remove(root_folder + 'Labels_' + j)
            os.remove(root_folder + 'Signal_' + j)


if __name__ == '__main__':
    path = '/srv/local/data/MGH_raw/train/'
    if not os.path.exists(path + 'processed/'):
        os.makedirs(path + 'processed/')
        os.makedirs(path + 'processed/pretext')
        os.makedirs(path + 'processed/train')
        os.makedirs(path + 'processed/test')
    file_name = os.listdir(path)
    label_file, signal_file = [], []
    for f in file_name:
        if f[:6] == 'Labels':
            label_file.append(f[7:])
        elif f[:6] == 'Signal':
            signal_file.append(f[7:])
    
    mutual_name = list(set(label_file).intersection(set(signal_file)))

    print (len(mutual_name))

    # label_file, signal_file = ['Labels_' + sub for sub in mutual_name], ['Signal_' + sub for sub in mutual_name]

    N, epoch_sec = 40, 30
    p_list = []
    for i in range(N):
        process = Process(target=train_val_test, args=(path, mutual_name, i, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()

