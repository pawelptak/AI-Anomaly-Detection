import pandas as pd
import numpy as np
import re
from urllib.parse import urlsplit, parse_qs
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import ast
from settings import *


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

def prepare_malicious_urls(url='./malicious_data/malicious_urls'):
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

def prepare_logs_dataframe(url='logs_parsed/nsmc-kibana_new.txt_structured.csv'):
    # check if values from logs are malicious
    df = pd.read_csv('logs_parsed/nsmc-kibana_new.txt_structured.csv')

    # add columns with default values
    df['url_malicious_score'] = 0
    df['label'] = 'Normal'
    df['size [B]'] = "0.0"
    df['time [ms]'] = "0"
    return df

def parse_time_and_size(df, parameter_list, idx):
    time = [x for x in parameter_list if 'ms ' in x]
    if time:
        time = re.search(r'([0-9]*)ms', time[0]).groups()[0]
        if time:
            df.loc[idx, 'time [ms]'] = time
    size = [x for x in parameter_list if 'B' in x]
    if len(size) > 1:
        size = size[-1]
    elif len(size) == 1:
        size = size[0]
    if size:
        size = re.search(r' (.*)B', size).groups()[0]
        if size:
            df.loc[idx, 'size [B]'] = size
    return df


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
                t = sorted([float(x) for x in tfidf_vectorizer_vectors.T.todense()], reverse=True)[:6]
                t = [x for x in t]
                score = sum(t)
                df.loc[idx, 'url_malicious_score'] = score
                if score > MALICIOUS_TRESHOLD:
                    df.loc[idx, 'label'] = 'Malicious'
            else:
                df.loc[idx, 'url_malicious_score'] = 0
            df = parse_time_and_size(df, parameter_list, idx)
    return df


def prepare_data():
    fitted_urls_vectorizer = prepare_tfidf_vectorizer(url='./malicious_data/malicious_urls')
    if not fitted_urls_vectorizer:
        raise Exception('No fitted vectorizer')
    df = prepare_logs_dataframe()
    df = calculate_malicious_score_in_df_urls(df, fitted_urls_vectorizer)
    df.to_csv('./logs_parsed/nsmc-kibana_new.txt_structured.csv', index=False)
    return df

if __name__ == '__main__':
    prepare_data()
