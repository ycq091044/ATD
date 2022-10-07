import os
import numpy as np
import pickle

np.random.seed(1234)

if not os.path.exists('./HAR/processed/'):
    os.makedirs('./HAR/processed/pretext')
    os.makedirs('./HAR/processed/train')
    os.makedirs('./HAR/processed/test')

# ------ test data ------------
test_X_folder = './HAR/test/Inertial Signals/'
test_y_folder = './HAR/test/y_test.txt'

signal_file = []

for name in os.listdir(test_X_folder):
    file_name = test_X_folder + name
    content = []
    for line in open(file_name, 'r').readlines():
        line_content = []
        # print (line[:-1].replace('  ', ' ').split(' '))
        # aa
        for number in line[:-1].replace('  ', ' ').split(' ')[1:]:
            # print (number)
            line_content.append(eval(number))
        content.append(line_content)
    signal_file.append(content)
    print ('finished', file_name)

X = np.array(signal_file) # (9, 7352, 128)

y = []

for line in open(test_y_folder, 'r').readlines():
    y.append(line[:-1])

train_index = np.random.choice(np.arange(2947), 2947 // 2, replace=False)

for index in train_index:
    path = './HAR/processed/train/sample-{}.pkl'.format(index)
    pickle.dump({'X': X[:, index, :], 'y': y[index]}, open(path, 'wb'))


test_index = set(np.arange(2947)) - set(list(train_index))

for index in test_index:
    path = './HAR/processed/test/sample-{}.pkl'.format(index)
    pickle.dump({'X': X[:, index, :], 'y': y[index]}, open(path, 'wb'))


# ------ training data ------------
train_X_folder = './HAR/train/Inertial Signals/'
train_y_folder = './HAR/train/y_train.txt'

signal_file = []

for name in os.listdir(train_X_folder):
    file_name = train_X_folder + name
    content = []
    for line in open(file_name, 'r').readlines():
        line_content = []
        # print (line[:-1].replace('  ', ' ').split(' '))
        # aa
        for number in line[:-1].replace('  ', ' ').split(' ')[1:]:
            # print (number)
            line_content.append(eval(number))
        content.append(line_content)
    signal_file.append(content)
    print ('finished', file_name)

X = np.array(signal_file) # (9, 7352, 128)

y = []

for line in open(train_y_folder, 'r').readlines():
    y.append(line[:-1])

for index in range(len(y)):
    path = './HAR/processed/pretext/sample-{}.pkl'.format(index)
    pickle.dump({'X': X[:, index, :], 'y': y[index]}, open(path, 'wb'))