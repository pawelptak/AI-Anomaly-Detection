import torch
from sklearn.model_selection import train_test_split
import numpy as np
from preparing.sliding_window_processor import prepare_dataframe, FeatureExtractor, fit_transform2
from preparing.prepare_dataset import prepare_custom_datasets, add_padding, logDataset
from system_log_parser import logs_parser
from settings import *
from preparing import prepare_data_in_parsed_file, prepare_model, standarize
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluate_after_training.plots import *
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
pd.options.mode.chained_assignment = None
import torch.nn.functional as F
import torch.nn as nn

def standarize_column(df, column):
    df[column] = (df[column] - df[column].mean()) / df[column].std()


def standarize_df(df_path, columns_to_standarize):
    df = pd.read_csv(df_path)
    for col in columns_to_standarize:
        standarize_column(df, col)
    df.to_csv(df_path)

PREPARE_DATA = True
if __name__ == '__main__':
    if PREPARE_DATA:
        # logs_parser.parser.parse(LOG_FILE_ALL)
        logs_prepared_df = prepare_data_in_parsed_file.prepare_data()
        x = logs_prepared_df.iloc[:]
        y = logs_prepared_df['label']
        re_pat = r"host=(\[.*])"

        # collecting events
        dataframe = prepare_dataframe(x, re_pat)
        # dataframe = standarize_df(df_path, columns_to_standarize=['url_malicious_score', 'time [ms]', 'size [B]'])
        x, labels = fit_transform2(dataframe)

        dataframe.to_csv("{}events.csv".format(LOGS_PARSED_OUTPUT_DIR), index=False)
        train_data, test_data, train_labels, test_labels = train_test_split(x, labels, test_size=0.2,
                                                                          random_state=42, stratify=labels)
        model = prepare_model.logCNN(2)
        model.to(DEVICE)
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        train_dataset = logDataset(train_data, labels=train_labels)
        test_dataset = logDataset(test_data, labels=test_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(NUM_EPOCHS):
            for i, (data, labels) in enumerate(train_loader):
                if data.shape[0] != BATCH_SIZE:
                    break
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print("Epoch: {}/{}".format(epoch, NUM_EPOCHS),
                          "Step: {}".format(i),
                          "Loss: {}".format(loss.item()))
                _, predicted = torch.max(outputs.data, 1)

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                if data.shape[0] != BATCH_SIZE:
                    break
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                not_correct = (predicted != labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        print('Not correct: %d' % not_correct)

        # test train data
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in train_loader:
                if data.shape[0] != BATCH_SIZE:
                    break
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                not_correct = (predicted != labels).sum().item()
        print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
        print('Not correct: %d' % not_correct)
        torch.save(model.state_dict(), 'model.pth')
