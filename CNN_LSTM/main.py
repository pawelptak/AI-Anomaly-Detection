import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from evaluation.metrics import test_model
from system_log_parser import logs_parser
from preparing.prepare_data import get_data_loaders, collect_events, fit_transform
from preparing import prepare_data, prepare_model, prepare_raw_nsmc_logs
from settings import *

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
pd.options.mode.chained_assignment = None


if __name__ == '__main__':
    if PREPARE_RAW_NSMC_LOGS:
        prepare_raw_nsmc_logs.prepare_raw_nsmc_data()

    if PARSE_LOGS:
        logs_parser.parser.parse(LOG_FILE_ALL)

    if PREPARE_DATAFRAME:
        logs_prepared_df = prepare_data.prepare_data()
    else:
        logs_prepared_df = prepare_data.load_prepared_data()

    x = logs_prepared_df.iloc[:]
    dataframe = collect_events(x)
    x, labels = fit_transform(dataframe)

    dataframe.to_csv("{}events.csv".format(LOGS_PARSED_OUTPUT_DIR), index=False)
    train_data, test_data, train_labels, test_labels = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                                        stratify=labels)
    model = prepare_model.logCNN(2)
    model.to(DEVICE)
    train_loader, test_loader = get_data_loaders(train_data, test_data, train_labels, test_labels)

    model = prepare_model.train_model(model, train_loader, DEVICE)

    test_model(test_loader, train_loader, model, DEVICE)
    torch.save(model.state_dict(), 'model.pth')
