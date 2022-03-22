import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from settings import *


class logDataset(Dataset):
    """Log Anomaly Features Dataset"""

    def __init__(self, data_vec, labels=None):
        self.X = data_vec
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data_matrix = self.X[idx]

        if not self.y is None:
            return (data_matrix, self.y[idx])
        else:
            return data_matrix


def add_padding(train_data, test_data, val_data):
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_data = F.pad(input=train_data, pad=(1, 1, 1, 1), mode='constant', value=0)  # pad all sides with 0s
    train_data = np.expand_dims(train_data, axis=1)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = F.pad(input=test_data, pad=(1, 1, 1, 1), mode='constant', value=0)
    test_data = np.expand_dims(test_data, axis=1)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    val_data = F.pad(input=val_data, pad=(1, 1, 1, 1), mode='constant', value=0)
    val_data = np.expand_dims(val_data, axis=1)
    return train_data, test_data, val_data


def prepare_custom_datasets(train_data, test_data, val_data, train_labels, test_labels, val_labels):
    # pass datasets into the custom dataclass
    train_dataset = logDataset(train_data, labels=train_labels)
    test_dataset = logDataset(test_data, labels=test_labels)
    val_dataset = logDataset(val_data, labels=val_labels)

    # use DataLoader class
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=0,
                              shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=0,
                             shuffle=False,
                             drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=0,
                            shuffle=False,
                            drop_last=True)

    return train_loader, test_loader, val_loader