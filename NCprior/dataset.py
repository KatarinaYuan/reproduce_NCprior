import os
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils import data as torch_data

from torchdrug import data, utils, core
from torchdrug.core import Registry as R

import ipdb

import numpy as np

@R.register("datasets.MNIST")
class MNIST(torch_data.Dataset, core.Configurable):

    """
    Classic MNIST dataset
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    num_samples = [50000, 10000, 10000]
    image_rows = 32
    image_cols = 32
    in_channels = 1

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        save_file = "mnist_%s" % os.path.basename(self.url)
        txt_file = os.path.join(path, save_file)
        if not os.path.exists(txt_file):
            txt_file = utils.download(self.url, self.path, save_file=save_file)
        
        self.load_npz_data(txt_file)

    def load_npz_data(self, file):

        with np.load(file) as f:
            x_train, y_train = f['x_train'], f['y_train'] # (60000, 28, 28), (60000,)
            x_test, y_test = f['x_test'], f['y_test'] # (10000, 28, 28), (10000,)
            
            x_train = np.pad(x_train, pad_width=((0,0), (2, 2), (2, 2)))
            x_test = np.pad(x_test, pad_width=((0,0), (2, 2), (2, 2)))
        
        self.data = np.concatenate((x_train, x_test), axis=0).astype(np.float32) / 255.0
        self.labels = np.concatenate((y_train, y_test), axis=0)

        assert self.image_rows == self.data.shape[1]
        assert self.image_cols == self.data.shape[2]

        self.data = torch.tensor(self.data).reshape(-1, 1, self.image_rows, self.image_cols)
        self.labels = torch.tensor(self.labels)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def split(self):

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, self.num_samples, generator=g)
