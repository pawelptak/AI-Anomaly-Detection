from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd


class UrlTFIDF:
    def __int__(self, url='./malicious_data_patterns/malicious_urls'):
        self.malicious_examples_url = url

    def get_tokens_for_tfidf(self, input):
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

    def prepare_malicious_urls(self):
        all_urls_csv = pd.read_csv(self.malicious_examples_url, header=None)
        data = all_urls_csv.values.tolist()
        random.shuffle(data)
        corpus = [str(d[0]) for d in data]
        return corpus

    def __init__(self, url):
        corpus = self.prepare_malicious_urls()
        if corpus:
            tfidf_vectorizer = TfidfVectorizer(tokenizer=self.get_tokens_for_tfidf)
            fitted_vectorizer = tfidf_vectorizer.fit(corpus)
            self.fitted_vectorizer = fitted_vectorizer
