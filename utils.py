import h5py
import torch
from torch.utils.data import Dataset

def display(mae, corr, multi_acc, bi_acc, f1):
    print('\nTest mae: {}'.format(mae))
    print('Test correlation: {}'.format(corr))
    print('Test multi-class accuracy: {}'.format(multi_acc))
    print('Test binary accuracy: {}'.format(bi_acc))
    print('Test f1 score: {}'.format(f1))

def total(params):
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings

def load_saved_data(input_dims):
    class LoadDataSet(Dataset):
        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __len__(self):
            return self.labels.shape[0]

        def __getitem__(self, idx):
            return self.audio[idx], self.visual[idx], self.text[idx], self.labels[idx]

    h5f = h5py.File('./data/X_train.h5', 'r')
    X_train = torch.from_numpy(h5f['data'][:]).to(torch.float32)
    h5f.close()
    h5f = h5py.File('./data/y_train.h5', 'r')
    y_train = torch.from_numpy(h5f['data'][:]).to(torch.float32)
    h5f.close()
    h5f = h5py.File('./data/X_valid.h5', 'r')
    X_valid = torch.from_numpy(h5f['data'][:]).to(torch.float32)
    h5f.close()
    h5f = h5py.File('./data/y_valid.h5', 'r')
    y_valid = torch.from_numpy(h5f['data'][:]).to(torch.float32)
    h5f.close()
    h5f = h5py.File('./data/X_test.h5', 'r')
    X_test = torch.from_numpy(h5f['data'][:]).to(torch.float32)
    h5f.close()
    h5f = h5py.File('./data/y_test.h5', 'r')
    y_test = torch.from_numpy(h5f['data'][:]).to(torch.float32)
    h5f.close()

    ad = input_dims[0]
    td = input_dims[2]
    train_data = LoadDataSet(X_train[:, :, td:td+ad], X_train[:, :, td+ad:], X_train[:, :, :td], y_train)
    valid_data = LoadDataSet(X_valid[:, :, td:td+ad], X_valid[:, :, td+ad:], X_valid[:, :, :td], y_valid)
    test_data = LoadDataSet(X_test[:, :, td:td+ad], X_test[:, :, td+ad:], X_test[:, :, :td], y_test)
    return train_data, valid_data, test_data
