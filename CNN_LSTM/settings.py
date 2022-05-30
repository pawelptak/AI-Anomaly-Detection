LOGS_FILE_RAW = "HDFS.log"
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

RAW_LOGS_INPUT_DIR = "data/logs_raw/"
LOGS_CSV_OUTPUT_DIR = "data/logs_prepared/"
LOGS_PARSED_OUTPUT_DIR = "data/logs_parsed/"
LOGS_INPUT_DIR = "data/logs_raw/"
LOGS_PREPARED_OUTPUT_DIR = "data/logs_prepared/"
PARSING_INPUT_DIR = "data/logs_prepared/"  # The input directory of log file
PARSING_OUTPUT_DIR = "data/logs_parsed/"  # The output directory of parsing results

# UNPARSED_FILE = "nsmc-kibana_unparsed_csv.txt"
UNPARSED_FILE = "k8s-dashboard-gk-keycloa.log"

PREPARE_RAW_NSMC_LOGS = False
PREPARE_DATAFRAME = False
PARSE_LOGS = False


LOG_FILE_ALL = f"{UNPARSED_FILE}"  # The input log file name
LOG_FORMAT = "<Date> <Type> <Content>"
PREPARED_DATA_URL = "./" + LOGS_PARSED_OUTPUT_DIR + "k8s-dashboard-gk-keycloa.log_structured.csv"
