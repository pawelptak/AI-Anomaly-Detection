import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from evaluation.metrics import TestProcessor
from system_log_parser.logs_parser import Parser
from preparing.prepare_model import train_model
from preparing.preparing import Preparing
from settings import *
from dataclasses import dataclass
pd.options.mode.chained_assignment = None


@dataclass
class Config:
    filename: str
    random_seed: int = 1
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 10
    num_classes: int = 2
    seq_len: int = 10
    num_layers: int = 1
    lstm_input_size: int = 32
    hidden_size: int = 20
    num_lstm_directions: int = 1
    malicious_treshold: float = 1.25
    raw_logs: bool = True
    parse_logs: bool = True
    prepare_dataframe: bool = True
    prepare_nsmc_logs_for_parsing: bool = True
    log_type: str = "nsmc"
    raw_logs_dir: str = "data/logs_raw/"
    prepared_logs_dir: str = "data/logs_prepared/"
    parsed_logs_dir: str = "data/logs_parsed/"
    prepared_data: str = None
    raw_data: str = None

    def __post_init__(self):
        self.prepared_data = "./" + self.parsed_logs_dir + self.filename + '_structured.csv'
        self.raw_data =  "./" + self.raw_logs_dir + self.filename


def main() -> None:
    config = Config(filename="nsmc-kibana-belk-kibana.log")
    preparing = Preparing(config)

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    else:
        DEVICE = "cpu"
    

    if config.prepare_nsmc_logs_for_parsing:
        preparing.prepare_raw_nsmc_logs_for_parsing()

    if config.parse_logs:
        parser = Parser(config)
        parser.parse_and_save_results()

    if config.prepare_dataframe:
        logs_prepared_df = preparing.preprocess_data()
    else:
        logs_prepared_df = preparing.load_preprocessed_data()

    # add const value
    logs_prepared_df['EventId'] = 'const'

    # logs_prepared_df = collect_events(x)
    x, labels = preparing.preprocessing.fit_transform(logs_prepared_df)

    logs_prepared_df.to_csv("{}events.csv".format(config.parsed_logs_dir), index=False)
    train_data, test_data, train_labels, test_labels = train_test_split(x, labels, test_size=0.2, random_state=42,
                                                                        stratify=labels)
    model = preparing.prepare_model()
    model.to(DEVICE)
    train_loader, test_loader = preparing.get_data_loaders(train_data, test_data, train_labels, test_labels)

    model = train_model(model, train_loader, DEVICE, config)


    test_processor = TestProcessor(config)
    test_processor.test_model(test_loader, train_loader, model, DEVICE)
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
