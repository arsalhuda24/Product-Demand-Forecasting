from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch


def sw(data, obs_len, pred_len):

    """implements a sliding window approach with window size = 1"""

    x1 = []
    y1 = []

    for i in range(len(data) - obs_len - 1):
        x = data[i:(i + obs_len)]
        y = data[(i + 1):(i + pred_len + 1)]

        x1.append(x)
        y1.append(y)
    return np.array(x1), np.array(y1)


class forecast(Dataset):

    """Custom dataset for LSTM input preperation
     The output of forecast class is [batch, seq_len, input_size]"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=12):
        super(forecast, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len

        path = os.path.join(self.data_dir, "train.csv")

        df = pd.read_csv(path)
        store = df[df.store == 1]
        self.df1 = pd.pivot_table(store, values="sales", index=["date", "store"], columns="item")
        self.df1 = self.df1.values

        a, b = sw(self.df1, self.obs_len, self.pred_len)
        self.a = a[0:1811]
        self.b = b[0:1811]

    def __len__(self):
        lenth = self.a.shape[0]
        return lenth

    def __getitem__(self, idx):
        #         print("A_index",self.a[idx])
        print("IDX", idx)
        out = [self.a[idx], self.b[idx]]
        return out


