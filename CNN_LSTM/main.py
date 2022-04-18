import torch
from sklearn.model_selection import train_test_split

from sliding_window_processor import prepare_dataframe, FeatureExtractor
from prepare_dataset import prepare_custom_datasets, add_padding
import prepare_model
import logs_parser
from settings import *
import prepare_data_in_parsed_file
from plots import *
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
