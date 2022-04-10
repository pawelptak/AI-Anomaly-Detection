import os
from settings import *
import re


def starts_with_timestamp(line):
    pattern = re.compile("^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    return bool(pattern.match(line))


def multiline_logs_processing(fpath):
    bracket_start = False
    bracket_log = ""
    outlines = []
    with open(fpath) as f:
        print(f'Processing {fpath}')
        lines = f.readlines()
        for line in lines:
            if starts_with_timestamp(line) and line.strip().endswith('{'):
                bracket_start = True
            if line.strip().startswith("}}''"):
                bracket_start = False
                bracket_log += line
                if len(bracket_log.strip()) > 2:
                    outlines.append(re.sub(r'\n(?=[^{}]*})', '', bracket_log))
                bracket_log = ""
            if starts_with_timestamp(line) and not line.strip().endswith('{'):
                outlines.append(line)

            if bracket_start:
                bracket_log += line.strip()

    out_name = os.path.splitext(os.path.basename(fpath))[0]
    with open(os.path.join(LOGS_CSV_OUTPUT_DIR, f'{out_name}.txt'), "w") as f:
        f.writelines(outlines)


if __name__ == '__main__':
    for fname in os.listdir(LOGS_INPUT_DIR):
        if fname.endswith('log'):
            fpath = os.path.join(LOGS_INPUT_DIR, fname)
            multiline_logs_processing(fpath)
