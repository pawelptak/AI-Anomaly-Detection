from settings import *

from sklearn.preprocessing import MinMaxScaler
import math
from collections import Counter
import pandas as pd
import numpy as np
import re
from urllib.parse import parse_qs

import ast
import multiprocessing
import time
from alive_progress import alive_bar


class Preprocessing:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def standarize_column(df, column):
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    def standarize_df(self, df_path, columns_to_standarize):
        df = pd.read_csv(df_path)
        for col in columns_to_standarize:
            self.standarize_column(df, col)
        df.to_csv(df_path)

    def collect_events(self, data_frame):
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

    def fit_transform(self, data_frame):
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
        with alive_bar(number_of_matrixes, title="Collecting windows for training...") as bar:
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
                bar()
        labels = np.array(labels)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        return x, labels

    def prepare_logs_dataframe(self):
        df = pd.read_csv(self.config.prepared_data)
        df['url_malicious_score'] = 0
        df['label'] = 'Normal'
        df['size [B]'] = "0.0"
        df['time [ms]'] = "0"
        return df


class K8sPreprocessing(Preprocessing):
    def __init__(self, config):
        super().__init__(config)

    def prepare_logs(self, df, fitted_vectorizer):
        pool = multiprocessing.Pool()
        start_time = time.perf_counter()
        with alive_bar(1, spinner='fishes', length=5, spinner_length=60, title="Preprocessing text logs...") as bar:
            processes = [pool.apply_async(self.parse_row, args=(row, fitted_vectorizer)) for _, row in df.iterrows()]
            result = [p.get() for p in processes]
        finish_time = time.perf_counter()
        print(f"Columns extracted in {finish_time - start_time} seconds")
        df = pd.DataFrame(result, columns=['size [B]', 'time [ms]', 'url_malicious_score', 'label'])
        print(df)
        return df

    def parse_row(self, row, fitted_vectorizer):
        size = 0
        time = 0
        label = 'Normal'
        url_malicious_score = 0
        parameter_list = row['ParameterList']
        parameter_list = ast.literal_eval(parameter_list)
        parameter_list = [n.strip() for n in parameter_list]
        for i, paramether in enumerate(parameter_list):
            if "path" in paramether:
                url = re.search(r'path\": "(.*)"', paramether).groups()[0]
                if url:
                    tfidf_vectorizer_vectors = fitted_vectorizer.fitted_vectorizer.transform([url])
                    t = sorted([float(x) for x in tfidf_vectorizer_vectors.T.todense()], reverse=True)[:10]
                    t = [1 / x for x in t if x != 0]
                    score = sum(t)
                    url_malicious_score = score
                    if score > self.config.malicious_treshold:
                        label = 'Malicious'
                else:
                    url_malicious_score = 0
            elif i == 0:
                try:
                    search_time = re.search(r'([0-9]*.[0-9]*),', paramether).groups()[0]
                    if search_time:
                        time = search_time
                except:
                    pass

            elif "bytes" in paramether:
                search_size = re.search(r': ([0-9]*),', paramether).groups()[0]
                if search_size:
                    size = search_size
        return size, time, url_malicious_score, label


class NsmcPreprocessing(Preprocessing):
    def __init__(self, config):
        super().__init__(config)

    def prepare_logs(self, df, fitted_vectorizer):
        for idx, row in df.iterrows():
            parameter_list = row['ParameterList']
            parameter_list = ast.literal_eval(parameter_list)
            parameter_list = [n.strip() for n in parameter_list]
            urls = [x for x in parameter_list if 'url' in x]
            for url in urls:
                url = re.search(r'\[(.*)\]', url).groups()[0]
                url = parse_qs(url)[''][0]
                if url:
                    tfidf_vectorizer_vectors = fitted_vectorizer.fitted_vectorizer.transform([url])
                    t = sorted([float(x) for x in tfidf_vectorizer_vectors.T.todense()], reverse=True)[:10]
                    t = [1 / x for x in t if x != 0]
                    score = sum(t)
                    df.loc[idx, 'url_malicious_score'] = score
                    if score > self.config.malicious_treshold:
                        df.loc[idx, 'label'] = 'Malicious'
                else:
                    df.loc[idx, 'url_malicious_score'] = 0
            size, time = self.parse_time_and_size(parameter_list)
            if size:
                df.loc[idx, 'size [B]'] = size
            if time:
                df.loc[idx, 'time [ms]'] = time
        return df

    def parse_time_and_size(self, parameter_list):
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
