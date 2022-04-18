import sys

from system_log_parser.Drain import Drain
from settings import *

regex = [
    r"host=(\[.*])",  # source
    r"url(=\[.*\])",  # url
    r"([0-9]*ms)",  # time
    r"([0-9]*\.[0-9]*B)",  # size
]
st = 0.3
depth = 9

parser = Drain.LogParser(
    LOG_FORMAT, indir=PARSING_INPUT_DIR, outdir=PARSING_OUTPUT_DIR, depth=depth, st=st, rex=regex
)
if __name__ == '__main__':
    parser.parse(LOG_FILE_ALL)
