# CNN anomaly detection


<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to detect anomalies from syslog/json data using CNN (Convolutional neural network)

The app will be deployed based on the following approaches:
* [Log Anomaly Detection](https://github.com/WraySmith/log-anomaly)
* [Using Machine Learning to Detect Malicious URLs](https://www.kdnuggets.com/2016/10/machine-learning-detect-malicious-urls.html)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

At this point, the project files should work on Windows and Linux with python 3.9.

### How to use this project?

1. Clone the repo
   ```
   git clone https://github.com/pawelptak/AI-Anomaly-Detection.git
   ```
2. Create virtualenv/venv and install requirements
    ```
   virtualenv anomaly_detection_env --python=3.9
   source anomaly_detection_env/bin/activate
   pip install -r requirements.txt
    ```
3. Copy file with raw logs into the RAW_LOGS_INPUT_DIR, change the name of the file to UNPARSED_FILE.
   
4. Run the app
    ```
   python main.py
   ```

## Custom options
You can modify what the app does by changing the following variables:

Run options:

 VARIABLE NAME          | DESCRIPTION
 --------------- | ------------------------
PREPARE_RAW_NSMC_LOGS   | If True, the raw syslog/json logs will be parsed and saved into a csv file
PARSE_LOGS              | If True, csv file will be parsed by Drain algorithm and saved in the LOGS_PARSED_OUTPUT_DIR directory
PREPARE_DATAFRAME       | If True, the parsed csv file will be transformed into a pandas dataframe that contains columns :Source host,EventId,url_malicious_score,time [ms],size [B],label. If False, the dataframe will be loaded from the LOGS_PARSED_OUTPUT_DIR directory



File options are:

 VARIABLE NAME      | DESCRIPTION
 --------------- | ------------------------
 RAW_LOGS_INPUT_DIR        |   Raw logs directory (json/syslog format)
 LOGS_CSV_OUTPUT_DIR      |   Directory that contains logs parsed to csv format (from json/syslog format)
 LOGS_PARSED_OUTPUT_DIR         | The output directory of parsing results (from csv format)
 UNPARSED_FILE         | File that contains raw logs in json/syslog format

Training options in current implementation:

 VARIABLE NAME      | DESCRIPTION
 --------------- | ------------------------
 RANDOM_SEED        |   Initialize a pseudo-random number sequence 
 LEARNING_RATE      |   parameter in an optimization algorithm that determines the step size at each iteration
 BATCH_SIZE         |   The number of training samples to work through before the model’s internal parameters are updated
 NUM_EPOCHS         |   Hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset  
 NUM_CLASSES        |   The number of classes in the classification problem    
 SEQ_LENGTH         |   The number of time steps that are used to predict the next time step
 NUM_LSTM_DIRECTIONS       |   The number of LSTM layers in the network 
 NUM_LAYERS         |   The number of LSTM layers in the network      
 LSTM_INPUT_SIZE    |   The number of features in the input sequence    
 HIDDEN_SIZE        |   The number of features in the hidden state of the LSTM layer
 MALICIOUS_TRESHOLD |   The threshold for the anomaly score

### How this project works?
CNN is able to detect anomalies in a stream of data. By now, it can detect Directory Traversal attack.
Here are the steps:

1. The raw logs are transformed to csv format (from json/syslog format)
2. The csv file is parsed by Drain algorithm and saved in the LOGS_PARSED_OUTPUT_DIR directory. Drain algorithm can parse syslog/json logs and extract features from them. In this case, the features are '''source, url, time and size'''.
3. The extracted features are transformed into a pandas dataframe that contains columns :Source host,EventId,url_malicious_score,time [ms],size [B],label. Url malicious score is calculated based on this approach: [Using Machine Learning to Detect Malicious URLs](https://www.kdnuggets.com/2016/10/machine-learning-detect-malicious-urls.html)
4. The dataframe is transformed into a sequence of time steps. Each time step is a row in the dataframe. The sequence length is defined by SEQ_LENGTH.
5. We train the model.
6. We test the model and display confusion matrix.

###TBC

