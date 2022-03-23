from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tqdm


class CICIDSPreprocessorWindows:
    def __init__(self, window_size=20, stride=10):
        self.to_delete_columns = ['Flow ID', ' Timestamp']
        self.label_column = ' Label'
        self.window_size = window_size
        self.stride = stride

    def preprocess_train_data(self, df: pd.DataFrame, label="BENIGN"):
        df = df.drop(self.to_delete_columns, axis=1)
        df = df[df[self.label_column] == label]
        df.reset_index(drop=True, inplace=True)
        df.drop(self.label_column, axis=1, inplace=True)

        df = df.fillna(0)

        return self.__get_windows(df, self.window_size, self.stride)

    def preprocess_test_data(self, df: pd.DataFrame, label="BENIGN"):
        df = df.drop(self.to_delete_columns, axis=1)

        status = pd.Series(
            [0 if i == label else 1 for i in df[self.label_column]])
        df.drop(self.label_column, axis=1, inplace=True)

        df = df.fillna(0)

        windows = self.__get_windows(df, self.window_size, self.stride)
        y_label = [1 if np.sum(status[i:i+self.window_size]) > 0 else 0 for i in range(
            0, len(df.values)-self.window_size+1, self.stride)]

        return(windows, y_label)

    def __get_windows(self, df, window_size=20, stride=10):
        windows_arr = []
        for i in tqdm.tqdm(range(0, len(df)-window_size+1, stride)):
            windows_arr.append(df.iloc[i:i+window_size, :].to_numpy())
        return np.array(windows_arr)
