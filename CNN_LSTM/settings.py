LOGS_FILE_RAW = "HDFS.log"
LOG_FORMAT = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_CLASSES = 2
SEQ_LENGTH = 10
NUM_LSTM_DIRECTIONS = 1
NUM_LAYERS = 1
LSTM_INPUT_SIZE = 32
HIDDEN_SIZE = 20
MALICIOUS_TRESHOLD = 1.25

RAW_LOGS_INPUT_DIR = "logs_data/logs_raw/"
LOGS_CSV_OUTPUT_DIR = "logs_data/logs_prepared/"
LOGS_PARSED_OUTPUT_DIR = "logs_data/logs_parsed/"
LOGS_INPUT_DIR = "logs_data/logs_raw/"
LOGS_PREPARED_OUTPUT_DIR = "logs_data/logs_prepared/"
PARSING_INPUT_DIR = "logs_data/logs_prepared/"  # The input directory of log file
PARSING_OUTPUT_DIR = "logs_data/logs_parsed/"  # The output directory of parsing results

UNPARSED_FILE = "nsmc-kibana_unparsed_csv.txt"

PREPARE_DATAFRAME = True
PARSE_LOGS = True
PREPARE_RAW_NSMC_LOGS = True

LOG_FILE_ALL = "nsmc-kibana_unparsed_csv.txt"  # The input log file name
LOG_FORMAT = "<Date> <Time> <Content>"
PREPARED_DATA_URL = "./" + LOGS_PARSED_OUTPUT_DIR + "nsmc-kibana_unparsed_csv.txt_structured.csv"
