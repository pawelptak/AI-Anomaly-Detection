import os
from settings import *

def multiline_logs_processing(fpath):
    bracket_start = False
    bracket_log = ""
    outlines = []
    with open(fpath) as f:
        print(f'Processing {fpath}')
        lines = f.readlines()
        for line in lines:
            if any(c.isalpha() for c in line) and not bracket_start and len(line) > 2 and ' => ' not in line:
                outlines.append(line)

            if line.startswith('{'):
                bracket_start = True
                bracket_log += line.strip()
                continue

            if line.startswith('}'):
                bracket_start = False
                bracket_log += line
                if len(bracket_log) > 2:
                    outlines.append(bracket_log)
                bracket_log = ""
                continue

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
