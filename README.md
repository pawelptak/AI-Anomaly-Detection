# AI Anomaly Detection 

Development of AI algorithms for anomaly detection in private telco cloud audit logs 

## LSTM-Autoencoder

## Convolutional Neural Network

## Dense-Autoencoder
Autoencoder with Dense layers for anomaly detection.

### Installation
Install all Python libraries listed in the dense_autoencoder/requirements.txt file.

### Processed data fromat
The script takes files with .csv format, with headers and comma separated data. The label column should be named 'Label', where 1 = anomaly, 0 = no anomaly. E.g.:
```
EventSequence,Label
191,0
203,1
199,0
278,0
```


### Usage
Run the scripts from terminal with necessary arguments.
#### Model training
```bash
python aytoencoder_training.py [processed data csv file path] [train data percentage]
```
Example:
```bash
python aytoencoder_training.py /data/processed/processed.csv 90
```

#### Model testing
```bash
python aytoencoder_testing.py [model .h5 file path]  -t[anomaly detection threshold] (optional)
```

Example:
```bash
python aytoencoder_testing.py dense_model.h5 -t 130
```

