from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tqdm

"""
Class for Preprocessing CICIDS2017 Data represented as rows
"""


class CICIDSPreprocessor:
    def __init__(self):
        self.to_delete_columns = ['Flow ID', ' Timestamp']
        self.label_column = ' Label'

    def preprocess_train_data(self, df: pd.DataFrame, label="BENIGN"):
        df = df.drop(self.to_delete_columns, axis=1)
        df = df[df[self.label_column] == label]
        df.reset_index(drop=True, inplace=True)
        df.drop(self.label_column, axis=1, inplace=True)

        return df.fillna(0)

    def preprocess_test_data(self, df: pd.DataFrame, label="BENIGN"):
        df = df.drop(self.to_delete_columns, axis=1)
        df = df[df[self.label_column] == label]
        df.reset_index(drop=True, inplace=True)
        df.drop(self.label_column, axis=1, inplace=True)

        return df.fillna(0)

    def __get_windows(self, df, window_size=20, stride=10):
        windows_arr = []
        for i in tqdm.tqdm(range(0, len(df)-window_size+1, stride)):
            windows_arr.append(df.iloc[i:i+window_size, :].to_numpy())
        return np.array(windows_arr)
