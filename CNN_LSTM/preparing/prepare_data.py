import torch
from settings import *
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import math
from collections import Counter
import pandas as pd
import numpy as np
import re
from urllib.parse import urlsplit, parse_qs
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import ast
from settings import PREPARED_DATA_URL


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

def standarize_column(df, column):
    df[column] = (df[column] - df[column].mean()) / df[column].std()


def standarize_df(df_path, columns_to_standarize):
    df = pd.read_csv(df_path)
    for col in columns_to_standarize:
        standarize_column(df, col)
    df.to_csv(df_path)


def get_data_loaders(train_data, test_data, train_labels, test_labels):
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = logDataset(train_data, labels=train_labels)
    test_dataset = logDataset(test_data, labels=test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader


def collect_events(data_frame):
    """
    turns input data_frame into a 2 columned dataframe
    with columns: BlockId, EventSequence
    where EventSequence is a list of the events that happened to the block
    """
    regex_pattern = r"host=(\[.*])"
    data_list = []
    for _, row in data_frame.iterrows():
        blk_id_list = re.findall(regex_pattern, row["Content"])
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            data_list.append({"Source host": str(blk_id),
                              "EventId": row["EventId"],
                              "url_malicious_score": float(row["url_malicious_score"]),
                              "time [ms]": float(row["time [ms]"]),
                              "size [B]": float(row["size [B]"]),
                              "label": str(row["label"])
                              })

    data_frame = pd.DataFrame(data_list)
    return data_frame


def fit_transform(data_frame):
    all_events = data_frame[["EventId"]].values
    all_events = all_events.reshape(-1)
    all_events_dict = dict(Counter(all_events))
    matrix_size = (6, 4)
    columns_in_one_log = 4
    logs_in_one_matrix = math.floor((matrix_size[0] * matrix_size[1]) / columns_in_one_log)
    number_of_all_matrixes = len(all_events) - 8
    data_frame["url_malicious_score"] = MinMaxScaler().fit_transform(
        data_frame["url_malicious_score"].values.reshape(-1, 1))
    data_frame["time [ms]"] = MinMaxScaler().fit_transform(data_frame["time [ms]"].values.reshape(-1, 1))
    data_frame["size [B]"] = MinMaxScaler().fit_transform(data_frame["size [B]"].values.reshape(-1, 1))
    rows_number = data_frame.shape[0]
    number_of_matrixes = rows_number - columns_in_one_log - 1
    labels = []
    x = np.zeros((number_of_matrixes, matrix_size[0], matrix_size[1]))
    for i in range(number_of_matrixes):
        window = data_frame.iloc[i:i + logs_in_one_matrix, :]
        window_labels = window.iloc[:, -1]
        features = window[["EventId", "url_malicious_score", "time [ms]", "size [B]"]]
        if features.shape[0] != 6:
            break
        events_in_window = dict(Counter(list(features.EventId.values)))
        for index, row in enumerate(features.itertuples()):
            features.iloc[index, 0] = math.log(number_of_all_matrixes / all_events_dict[row.EventId])
        features_numpy = np.array(features.values)
        if np.any(window_labels == "Malicious"):
            labels.append(1)
        else:
            labels.append(0)
        x[i] = features_numpy
    labels = np.array(labels)
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    return x, labels



def get_tokens_for_tfidf(input):
    tokens_by_slash = str(input.encode('utf-8')).split('/')
    tokens_by_slash += str(input.encode('utf-8')).split('\\')
    all_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')
        tokens_by_dot = []
        for j in range(len(tokens)):
            temp_tokens = str(tokens[j]).split('.')
            tokens_by_dot += temp_tokens
        all_tokens += tokens + tokens_by_dot
    all_tokens += tokens_by_slash
    all_tokens = list(set(all_tokens))
    if 'com' in all_tokens:
        all_tokens.remove('com')
    if 'pl' in all_tokens:
        all_tokens.remove('pl')
    return all_tokens

def prepare_malicious_urls(url='./malicious_data_patterns/malicious_urls'):
    all_urls_csv = pd.read_csv(url, header=None)
    data = all_urls_csv.values.tolist()
    random.shuffle(data)
    corpus = [str(d[0]) for d in data]
    return corpus

def prepare_tfidf_vectorizer(url):
    corpus = prepare_malicious_urls(url)
    if corpus:
        tfidf_vectorizer = TfidfVectorizer(tokenizer=get_tokens_for_tfidf)
        fitted_vectorizer = tfidf_vectorizer.fit(corpus)
        return fitted_vectorizer

def prepare_logs_dataframe(url='./logs_data/logs_parsed/nsmc-kibana_new.txt_structured.csv'):
    df = pd.read_csv(PREPARED_DATA_URL)

    df['url_malicious_score'] = 0
    df['label'] = 'Normal'
    df['size [B]'] = "0.0"
    df['time [ms]'] = "0"
    return df

def parse_time_and_size(df, parameter_list, idx):
    time = [x for x in parameter_list if 'ms ' in x]
    if time:
        time = re.search(r'([0-9]*)ms', time[0]).groups()[0]
    size = [x for x in parameter_list if 'B' in x]
    if len(size) > 1:
        size = size[-1]
    elif len(size) == 1:
        size = size[0]
    if size:
        size = re.search(r' (.*)B', size).groups()[0]
    return size, time


def calculate_malicious_score_in_df_urls(df, fitted_vectorizer):
    for idx, row in df.iterrows():
        parameter_list = row['ParameterList']
        parameter_list = ast.literal_eval(parameter_list)
        parameter_list = [n.strip() for n in parameter_list]
        urls = [x for x in parameter_list if 'url' in x]
        for url in urls:
            url = re.search(r'\[(.*)\]', url).groups()[0]
            url = parse_qs(url)[''][0]
            if url:
                tfidf_vectorizer_vectors = fitted_vectorizer.transform([url])
                t = sorted([float(x) for x in tfidf_vectorizer_vectors.T.todense()], reverse=True)[:10]
                t = [1/x for x in t if x != 0]
                score = sum(t)
                df.loc[idx, 'url_malicious_score'] = score
                if score > MALICIOUS_TRESHOLD:
                    df.loc[idx, 'label'] = 'Malicious'
            else:
                df.loc[idx, 'url_malicious_score'] = 0
        size, time = parse_time_and_size(df, parameter_list, idx)
        if size:
            df.loc[idx, 'size [B]'] = size
        if time:
            df.loc[idx, 'time [ms]'] = time
    return df


def prepare_data():
    fitted_urls_vectorizer = prepare_tfidf_vectorizer(url='./malicious_data_patterns/malicious_urls')
    if not fitted_urls_vectorizer:
        raise Exception('No fitted vectorizer')
    df = prepare_logs_dataframe()
    df = calculate_malicious_score_in_df_urls(df, fitted_urls_vectorizer)
    df.to_csv(PREPARED_DATA_URL, index=False)
    return df

def load_prepared_data():
    return pd.read_csv(PREPARED_DATA_URL)

if __name__ == '__main__':
    prepare_data()
