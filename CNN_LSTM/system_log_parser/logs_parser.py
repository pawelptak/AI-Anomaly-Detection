import sys

from system_log_parser.Drain import Drain
from settings import *

# regex = [
#     r"host=(\[.*])",  # source
#     r"url(=\[.*\])",  # url
#     r"([0-9]*ms)",  # time
#     r"([0-9]*\.[0-9]*B)",  # size
# ]
#
# regex = [
#     r'client_ip\": "(.*)[0-9]",',  # source
#     r'path\": "(.*)"',  # url
#     r'method\": "(.*)",',  # method
#     r'bytes\": [0-9]*,',  # size
#     r'status\": [0-9]*,',  # status
#     r'latency\": ([0-9]*.[0-9]*),',  # time
# ]

regex = [
    r'(client_ip\": ".*[0-9])",',  # source
    r'(path\": ".*)"',  # url
    r'(method\": "(.*))",',  # method
    r'(bytes\": [0-9]*),',  # size
    r'(status\": [0-9])*,',  # status
    r'{(latency\": [0-9]*.[0-9]*),',  # time
]
st = 0.3
depth = 9

parser = Drain.LogParser(
    LOG_FORMAT, indir=PARSING_INPUT_DIR, outdir=PARSING_OUTPUT_DIR, depth=depth, st=st, rex=regex
)
if __name__ == '__main__':
    parser.parse(LOG_FILE_ALL)
