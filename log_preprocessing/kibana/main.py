import json
import os

import pandas as pd
from pandas import json_normalize

from settings import *


def json_logs_to_csv(fpath):
    dfs = []
    with open(fpath) as f:
        logs = f.readlines()
        for log in logs:
            json_log = json.loads(log)
            df = json_normalize(json_log)
            dfs.append(df)
    all_logs_df = pd.concat(dfs)
    all_logs_df.dropna(how='all', axis=1, inplace=True)
    all_logs_df.to_csv(f'{os.path.join(LOGS_CSV_OUTPUT_DIR, os.path.splitext(os.path.basename(fpath))[0])}.csv', index=False)


if __name__ == '__main__':
    for fname in os.listdir(LOGS_INPUT_DIR):
        if fname.endswith('log'):
            fpath = os.path.join(LOGS_INPUT_DIR, fname)
            json_logs_to_csv(fpath)
