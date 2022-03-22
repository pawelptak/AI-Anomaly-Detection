import sys

from logparser.Drain import Drain
from settings import *

regex = [
    r"blk_(|-)[0-9]+",  # block id
    r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP
    r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # Numbers
]
st = 0.5
depth = 4

parser = Drain.LogParser(
    LOG_FORMAT, indir=LOGS_INPUT_DIR, outdir=LOGS_PARSED_OUTPUT_DIR, depth=depth, st=st, rex=regex
)


def parse_logs(logs_path):
    parser.parse(logs_path)
