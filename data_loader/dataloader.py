import numpy as np
from spektral.data import Dataset
from spektral.data.graph import Graph

class TrafficDataset(Dataset):
    def __init__(self, config,data, W_mat, **kwargs):
        self.config=config
        self.data = data
        self.W_mat = W_mat
        super().__init__(**kwargs)

    def read(self):
        def make_graph(i, j):
            sta = i * self.config['N_DAY_SLOT'] + j
            n_window = self.config['N_PRED'] + self.config['N_HIST']
            end = sta + n_window
            #             print(sta,end)
            full_window = np.swapaxes(self.data[sta:end, :], 0, 1)
            x = full_window[:, 0:self.config['N_HIST']]
            y = full_window[:, self.config['N_HIST']::]
            a = self.W_mat
            return Graph(x=x, a=a, y=y)

        return [make_graph(i, j) for i in range(self.config['N_DAYS']) for j in range(self.config['N_SLOT'])]


def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=True):
    """"
    Given distances between all nodes, convert into a weight matrix
    :param W distances
    :param sigma2 User configurable parameter to adjust sparsity of matrix
    :param epsilon User configurable parameter to adjust sparsity of matrix
    :param gat_version If true, use 0/1 weights with self loops. Otherwise, use float
    """
    n = W.shape[0]
    W = W / 10000.
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    # refer to Eq.10
    W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

    # If using the gat version of this, round to 0/1 and include self loops
    if gat_version:
        W[W>0] = 1
        W += np.identity(n)

    return W


def get_splits(dataset, n_slot, splits):
    """
    Given the data, split it into random subsets of train, val, and test as given by splits
    :param dataset: TrafficDataset object to split
    :param n_slot: Number of possible sliding windows in a day
    :param splits: (train, val, test) ratios
    """
    split_train, split_val, _ = splits
    i = n_slot*split_train
    j = n_slot*split_val
    train = dataset[:i]
    val = dataset[i:i+j]
    test = dataset[i+j:]

    return train, val, test