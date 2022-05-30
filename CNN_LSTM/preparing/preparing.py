from prepare_raw_nsmc_logs import PrepareNSMCLogs
from prepare_model import logCNN
from tfidf import UrlTFIDF
from prepare_data import NsmcPreprocessing, K8sPreprocessing
import torch
from data.dataset import logDataset
from torch.utils.data import DataLoader
import pandas as pd



class Preparing:
    preprocessing = None

    def __init__(self, config):
        self.config = config

    def prepare_raw_nsmc_logs_for_parsing(self):
        prepare_nsmc_logs = PrepareNSMCLogs(self.config)
        df = prepare_nsmc_logs.prepare_raw_nsmc_data()
        prepare_nsmc_logs.save_prepared_data(df)

    def prepare_model(self):
        model = logCNN(self.config)
        return model

    def preprocess_data(self):
        fitted_urls_vectorizer = UrlTFIDF(url='./malicious_data_patterns/malicious_urls')
        if not fitted_urls_vectorizer:
            raise Exception('No fitted vectorizer')

        if self.config.log_type == 'nsmc':
            self.preprocessing = NsmcPreprocessing(self.config)
        elif self.config.log_type == 'k8s':
            self.preprocessing = K8sPreprocessing(self.config)
        else:
            raise Exception('No such log type')
        df = self.preprocessing.prepare_logs_dataframe()
        df = self.preprocessing.prepare_logs(df, fitted_urls_vectorizer)
        df.to_csv(self.config.prepared_logs_dir+, index=False)
        return df

    def load_preprocessed_data(self):
        return pd.read_csv(self.config.prepared_data)

    def get_data_loaders(self, train_data, test_data, train_labels, test_labels):
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        train_dataset = logDataset(train_data, labels=train_labels)
        test_dataset = logDataset(test_data, labels=test_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=True)
        return train_loader, test_loader
