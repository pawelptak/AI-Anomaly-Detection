import os
from settings import *
import re

def multiline_logs_processing(fpath):
    bracket_start = False
    bracket_log = ""
    outlines = []
    with open(fpath) as f:
        print(f'Processing {fpath}')
        lines = f.readlines()
        for line in lines:
            if any(c.isalpha() for c in line) and not bracket_start and len(line) > 2 and ' => ' not in line and not line.strip().endswith('{'):
                outlines.append(line)

            if line.strip().endswith('{'):
                bracket_start = True

            if line.strip().endswith('}'):
                bracket_start = False
                bracket_log += line
                if len(bracket_log.strip()) > 2:
                    outlines.append(re.sub(r'\n(?=[^{}]*})', '', bracket_log))
                bracket_log = ""

            if bracket_start:
                bracket_log += re.sub(r'\n(?=[^{}]*})', '', line.strip())

    out_name = os.path.splitext(os.path.basename(fpath))[0]
    with open(os.path.join(LOGS_CSV_OUTPUT_DIR, f'{out_name}.txt'), "w") as f:
        f.writelines(outlines)


if __name__ == '__main__':
    for fname in os.listdir(LOGS_INPUT_DIR):
        if fname.endswith('log'):
            fpath = os.path.join(LOGS_INPUT_DIR, fname)
            multiline_logs_processing(fpath)
