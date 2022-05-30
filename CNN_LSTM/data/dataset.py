from torch.utils.data import Dataset


class logDataset(Dataset):

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
