import numpy as np
import pandas as pd
from glob import glob
import os


class min_max_normalization:

    def __init__(self, values, min_max_norm=None):
        if min_max_norm is None:
            self.min_val = np.min(values)
            self.max_val = np.max(values)
        else:
            self.min_val = min_max_norm.min_val
            self.max_val = min_max_norm.max_val
        self.normalized = np.array((values - self.min_val) / (self.max_val - self.min_val))

    def print_min_max(self):
        print("Min value:")
        print(self.min_val)
        print("Max value:")
        print(self.max_val)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def extract_params(file_name):
    file_name = os.path.basename(os.path.splitext(file_name)[0])
    params = file_name.split('_')
    return np.array([float(params[1]), float(params[3]), float(params[5]), float(params[7])])


def create_dataframe(path, df_name):
    files = glob(path)

    params = np.array(list(map(extract_params, files)))

    min_max = [min_max_normalization(params[:, 0]), min_max_normalization(params[:, 1]),
               min_max_normalization(params[:, 2]), min_max_normalization(params[:, 3])]

    data = {'path': files, 'g': sigmoid(min_max[0].normalized), 'amp': sigmoid(min_max[1].normalized),
               'at': sigmoid(min_max[2].normalized), 'time': sigmoid(min_max[3].normalized)}

    df = pd.DataFrame(data=data)

    if df_name is not None:
        df.to_csv(df_name, index=False)
    return df, min_max



def create_train_valid_dataframe(train_path, valid_path):
    train_df, train_min_max = create_dataframe(train_path, './train_df.csv')
    valid_df, valid_min_max = create_dataframe(valid_path, './valid_df.csv')

    return train_df, valid_df, train_min_max, valid_min_max


def params_names_to_indices(params_names):
    names_to_indices = {"g": 0, "amp": 1, "at": 2, "time": 3}
    params_indices = []
    for name in params_names:
        params_indices.append(names_to_indices[name])
    return params_indices