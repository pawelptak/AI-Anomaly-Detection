import torch
from sklearn.model_selection import train_test_split

from preparing.sliding_window_processor import prepare_dataframe, FeatureExtractor
from preparing.prepare_dataset import prepare_custom_datasets, add_padding
from system_log_parser import logs_parser
from settings import *
from preparing import prepare_data_in_parsed_file, prepare_model, standarize
from evaluate_after_training.plots import *
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

PREPARE_DATA = True
if __name__ == '__main__':
    if PREPARE_DATA:
        logs_parser.parser.parse(LOG_FILE_ALL)
        logs_prepared_df = prepare_data_in_parsed_file.prepare_data()
        x = logs_prepared_df.iloc[:]
        y = logs_prepared_df['label']
        re_pat = r"host=(\[.*])"

        # collecting events
        dataframe = prepare_dataframe(x, re_pat)
        dataframe.to_csv("{}events.csv".format(LOGS_PARSED_OUTPUT_DIR), index=False)
        standarize.standarize_df("{}events.csv".format(LOGS_PARSED_OUTPUT_DIR), columns_to_standarize=['url_malicious_score', 'time [ms]', 'size [B]'])
