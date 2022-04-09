import os
from settings import *

def multiline_logs_processing(fpath):
    bracket_start = False
    brackets = []
    bracket_log = ""
    outlines = []
    with open(fpath) as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('{') and not bracket_start and ' => ' not in line:
                outlines.append(line)

            if line.startswith('{'):
                bracket_start = True
                bracket_log += line.strip()
                continue

            if line.startswith('}'):
                bracket_start = False
                bracket_log += line
                outlines.append(bracket_log)
                bracket_log = ""
                continue

            if bracket_start:
                bracket_log += line.strip()

    print(brackets)
    out_name = os.path.splitext(os.path.basename(fpath))[0]
    with open(os.path.join(LOGS_CSV_OUTPUT_DIR, 'pies.txt'), "w") as f:
        f.writelines(outlines)
            #print(bracket_log)


if __name__ == '__main__':
    for fname in os.listdir(LOGS_INPUT_DIR):
        if fname.endswith('log'):
            fpath = os.path.join(LOGS_INPUT_DIR, fname)
            multiline_logs_processing(fpath)
            break
