import sys

from system_log_parser.Drain import Drain

class Parser:
    # This class prepares and run Drain parser
    st = 0.3
    depth = 9
    log_format = "<Date> <Type> <Content>"
    
    def __init__(self, config):
        if config.log_type == "nsmc":
            self.regex = [
                r'client_ip\": "(.*)[0-9]",',  # source
                r'path\": "(.*)"',  # url
                r'method\": "(.*)",',  # method
                r'bytes\": [0-9]*,',  # size
                r'status\": [0-9]*,',  # status
                r'latency\": ([0-9]*.[0-9]*),',  # time
            ]
        elif config.log_type == "k8s":
            self.regex = [
                r'(client_ip\": ".*[0-9])",',  # source
                r'(path\": ".*)"',  # url
                r'(method\": "(.*))",',  # method
                r'(bytes\": [0-9]*),',  # size
                r'(status\": [0-9])*,',  # status
                r'{(latency\": [0-9]*.[0-9]*),',  # time
            ]
        
        self.parser = Drain.LogParser(
            self.log_format, indir=config.raw_logs_dir, outdir=config.parsed_logs_dir, depth=self.depth, st=self.st, rex=self.regex
        )
        self.filename = config.filename

    def parse_and_save_results(self):
        self.parser.parse(self.filename)

