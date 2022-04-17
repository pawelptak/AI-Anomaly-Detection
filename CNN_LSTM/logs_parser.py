import sys

from logparser.Drain import Drain
from settings import *

input_dir = "logs_prepared/"  # The input directory of log file
output_dir = "logs_parsed/"  # The output directory of parsing results
log_file_all = "nsmc-kibana_new.txt"  # The input log file name
log_format = "<Date> <Time> <Content>"
regex = [
    r"host=(\[.*])",  # source
    r"url(=\[.*\])",  # url
    r"([0-9]*ms)",  # time
    r"([0-9]*\.[0-9]*B)",  # size
]
st = 0.8
depth = 9

parser = Drain.LogParser(
    log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex
)
if __name__ == '__main__':
    parser.parse(log_file_all)
