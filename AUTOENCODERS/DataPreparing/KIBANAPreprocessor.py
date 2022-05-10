import enum
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tqdm
from datetime import datetime
import re
import pickle
from urllib.parse import parse_qs

def calculate_malucious_score_for_urls(df):
    df.insert(loc=0, column='url_mal_score', value=np.zeros(len(df)))
    vecorizer = pickle.load(open('vectorizer.pk', 'rb'))
    for i, value in enumerate(df['req.url']):
        if value and type(value) == str:
            url = parse_qs(value)
            for key, value in url.items():
                tfidf_vectorizer_vectors = vecorizer.transform([value[0]])
                score = sum([float(x) for x in tfidf_vectorizer_vectors.T.todense()])
                if score > 0:
                    df.loc[i, 'url_mal_score'] += score
    return df

class KIBANAPreprocessor:
    def __init__(self, windows=False, windows_size=20, windows_stride=2):
        self.columns_to_keep = ['type', 'method',
                                'statusCode', 'time_gap', 'req.url']  # , 'message', 'req.url']
        self.categorical_columns = ['type', 'method', 'statusCode']
        self.windows_size = windows_size
        self.windows_stride = windows_stride
        self.windows = windows

    def preprocess_train_data(self, df: pd.DataFrame):
        # df = self.append_prev_request_time(df)
        df = df.filter(self.columns_to_keep)
        df = df.astype({col: str for col in self.categorical_columns})

        self.train_categorical_unique = {col: np.array(sorted(
            df[col].unique())) for col in self.categorical_columns}

        self.onehotencoders = {col: OneHotEncoder(
            handle_unknown='ignore') for col in self.categorical_columns}

        for (col, encoder) in self.onehotencoders.items():
            encoder.fit(self.train_categorical_unique[col].reshape(-1, 1))

            encoded_columns = [
                f'{col}_{c}' for c in self.train_categorical_unique[col]]
            new_data = encoder.transform(
                df[col].values.reshape(-1, 1)).toarray()
            new_data_frame = pd.DataFrame(new_data, columns=encoded_columns)
            df = df.join(new_data_frame)
            df.drop(col, axis=1, inplace=True)

        df = calculate_malucious_score_for_urls(df)
        df.drop('req.url', axis=1, inplace=True)

        # self.scaler = MinMaxScaler()
        # x = df.values  # returns a numpy array
        # df = pd.DataFrame(self.scaler.fit_transform(x), columns=df.columns)

        if(not self.windows):
            return df
        return self.__get_windows(df)

    def preprocess_test_data(self, df: pd.DataFrame):
        # df = self.append_prev_request_time(df)
        df = df.filter(self.columns_to_keep)
        df = df.astype({col: str for col in self.categorical_columns})

        for (col, encoder) in self.onehotencoders.items():
            encoded_columns = [
                f'{col}_{c}' for c in self.train_categorical_unique[col]]
            new_data = encoder.transform(
                df[col].values.reshape(-1, 1)).toarray()
            new_data_frame = pd.DataFrame(new_data, columns=encoded_columns)
            df = df.join(new_data_frame)
            df.drop(col, axis=1, inplace=True)

        df = calculate_malucious_score_for_urls(df)
        df.drop('req.url', axis=1, inplace=True)
        if(not self.windows):
            return df
        return self.__get_windows(df)

    def append_prev_request_time(self, df: pd.DataFrame):
        time = df['@timestamp']
        time = [datetime.fromisoformat(t[:-1]).timestamp() for t in time]
        time_gap = [t - time[i-1] for i, t in enumerate(time)]
        time_gap[0] = np.mean(time_gap)
        df.insert(loc=0, column='time_gap', value=time_gap)
        return df

    def __get_windows(self, df):
        windows_arr = []
        for i in tqdm.tqdm(range(0, len(df)-self.windows_size+1, self.windows_stride)):
            windows_arr.append(df.iloc[i:i+self.windows_size, :].to_numpy())
        return np.array(windows_arr)
