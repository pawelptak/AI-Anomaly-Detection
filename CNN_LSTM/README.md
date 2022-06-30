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
You can modify what the app does by changing the following variables in the Config class:

Available options:

 VARIABLE NAME          | DESCRIPTION
 --------------- | ------------------------
 filename | File with logs
prepare_nsmc_logs_for_parsing   | If True, the raw syslog/json logs will be parsed and saved into a csv file
parse_logs              | If True, csv file will be parsed by Drain algorithm and saved in the parsed_logs_dir directory
prepared_data       | If True, the parsed csv file will be transformed into a pandas dataframe that contains columns :Source host,EventId,url_malicious_score,time [ms],size [B],label. If False, the dataframe will be loaded from the parsed_logs_dir directory
 raw_logs_dir        |   Raw logs directory (json/syslog format)
 prepared_logs_dir      |   Directory that contains logs parsed to csv format (from json/syslog format)
 parsed_logs_dir         | The output directory of parsing results (from csv format)

Training options in current implementation:

 VARIABLE NAME      | DESCRIPTION
 --------------- | ------------------------
 random_seed        |   Initialize a pseudo-random number sequence 
 learning_rate      |   parameter in an optimization algorithm that determines the step size at each iteration
 batch_size         |   The number of training samples to work through before the model’s internal parameters are updated
 num_epochs         |   Hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset  
 num_classes        |   The number of classes in the classification problem    
 seq_len         |   The number of time steps that are used to predict the next time step
 num_lstm_directions       |   The number of LSTM layers in the network 
 num_layers         |   The number of LSTM layers in the network      
 lstm_input_size    |   The number of features in the input sequence    
 hidden_size        |   The number of features in the hidden state of the LSTM layer
 malicious_treshold |   The threshold for the anomaly score

### How this project works?
CNN is able to detect anomalies in a stream of data. By now, it can detect Directory Traversal attack.
Here are the steps:

1. The raw logs are transformed to csv format (from json/syslog format)
2. The csv file is parsed by Drain algorithm and saved in the parsed_logs_dir directory. Drain algorithm can parse syslog/json logs and extract features from them. In this case, the features are '''source, url, time and size'''.
3. The extracted features are transformed into a pandas dataframe that contains columns :Source host,EventId,url_malicious_score,time [ms],size [B],label. Url malicious score is calculated based on this approach: [Using Machine Learning to Detect Malicious URLs](https://www.kdnuggets.com/2016/10/machine-learning-detect-malicious-urls.html)
4. The dataframe is transformed into a sequence of time steps. Each time step is a row in the dataframe. The sequence length is defined by seq_len.
5. We train the model.
6. We test the model and display confusion matrix.

###Important URLs

1. Web attacks generator https://github.com/Rovlet/LogsGenerator
2. First approach to detect web attacks using CNN https://github.com/Rovlet/CNN_anomaly_detection


