# Logstash logs preprocessing
Converting logstash pod logs from .log file with multiline curly bracket lines to .txt file with single line curly bracket lines.

### Tested log files
connector-logstash-lannion-nacm-6c5bdf875b-cx4rt.log

connector-logstash-lannion-qualys-694799dd64-7xs48.log

### Usage
Put raw log files in the logs_raw directory. 

Run the main.py script. 

Output logs will be in the logs_processed directory.
