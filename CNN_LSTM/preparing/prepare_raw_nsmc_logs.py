import pandas as pd
import numpy as np
import re
import os
from settings import RAW_LOGS_INPUT_DIR, LOGS_CSV_OUTPUT_DIR, UNPARSED_FILE, LOGS_PARSED_OUTPUT_DIR
from pandas import json_normalize
import json

def starts_with_timestamp(line):
    pattern = re.compile("^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    return bool(pattern.match(line))


def multiline_logs_processing(fpath):
    dfs = []
    with open(fpath) as f:
        logs = f.readlines()
        for log in logs:
            json_log = json.loads(log)
            df = json_normalize(json_log)
            dfs.append(df)
    all_logs_df = pd.concat(dfs)
    all_logs_df.dropna(how='all', axis=1, inplace=True)
    all_logs_df.to_csv(f'{LOGS_PARSED_OUTPUT_DIR}{UNPARSED_FILE}', index=False)
    return all_logs_df


def prepare_raw_nsmc_data():
    for fname in os.listdir(RAW_LOGS_INPUT_DIR):
        if fname.endswith('log'):
            fpath = os.path.join(RAW_LOGS_INPUT_DIR, fname)
            df = multiline_logs_processing(fpath)

    print("Logs are prepared in csv format and saved to: ", LOGS_PARSED_OUTPUT_DIR)
    df = pd.read_csv(f'{LOGS_PARSED_OUTPUT_DIR}{UNPARSED_FILE}')
    df = df.drop(['type', 'tags', 'pid', 'method', 'statusCode', 'req.url', 'req.method', 'res.responseTime',
                  'req.headers.accept', 'req.remoteAddress', 'req.userAgent', 'res.statusCode', 'res.contentLength',
                  'req.headers.x-request-id', 'req.headers.x-real-ip', 'req.headers.x-forwarded-for',
                  'req.headers.x-forwarded-host', 'req.headers.x-forwarded-proto', 'req.headers.x-original-uri',
                  'req.headers.x-scheme', 'req.headers.content-length', 'req.headers.accept-language',
                  'req.headers.accept-encoding', 'req.headers.kbn-version', 'req.headers.origin',
                  'req.headers.referer', 'req.headers.sec-fetch-dest', 'req.headers.sec-fetch-mode',
                  'req.headers.sec-fetch-site', 'req.headers.netguard-proxy-roles', 'req.headers.username',
                  'req.referer', 'req.headers.content-type', 'req.headers.sec-ch-ua', 'req.headers.sec-ch-ua-mobile',
                  'req.headers.sec-ch-ua-platform', 'req.headers.upgrade-insecure-requests',
                  'req.headers.sec-fetch-user', 'req.headers.x-requested-with', 'req.headers.cache-control',
                  'state', 'prevState', 'prevMsg', 'req.headers.if-none-match', 'req.headers.if-modified-since',
                  'req.headers.dnt', 'req.headers.kbn-xsrf'], axis=1)

    # remove some special signs from column names
    df.columns = [re.sub('\.', '_', col) for col in df.columns]
    df.columns = [re.sub('-', '_', col) for col in df.columns]
    df.columns = [re.sub('@', '', col) for col in df.columns]

    for idx, row in df.iterrows():
        message_groups = re.match(r'^(\w*) /(\w*)/(\w*)/(\w*).=(.+) ([0-9][0-9][0-9]) ([0-9]+ms) - ([0-9]+\.[0-9]+B)', row.message)
        '''
        this regex if for parsing data like this: 
        POST /api/saved_objects/_bulk_get?=%2Fvar%2Flib%2Fmlocate.db 200 6ms - 9.0B
        '''
        if message_groups:
            message_groups = list(message_groups.groups())
            url = message_groups[4]
            url = f'url=[={url}]'
            message_groups[4] = url
            df.loc[idx, 'message'] = ' '.join(message_groups)
        else:
            message_groups = re.match(r'^(\w*) /(\w*)/(\w*).*([0-9][0-9][0-9]) ([0-9]+ms) - ([0-9]+\.[0-9]+B)', row.message)
            '''
            this regex if for parsing data like this: 
            GET /api/status?pretty= 200 8ms - 9.0B
            '''
            if message_groups is None:
                ''' this is for even shorter messages '''
                message_groups = re.match(r'^(\w*).*([0-9][0-9][0-9]) ([0-9]+ms) - ([0-9]+\.[0-9]+B)', row.message)
            if message_groups:
                message_groups = list(message_groups.groups())
                # if match, change the message, if not, leave it as it is
                df.loc[idx, 'message'] = ' '.join(message_groups)

        # change host value to format that is easy to parse
        host = row.req_headers_host
        host = f'host=[{host}]'
        df.loc[idx, 'req_headers_host'] = host

        # change req_headers_user_agent values to popular services names
        req_headers_user_agent_groups = re.match(r'^(\w*)/', str(row.req_headers_user_agent))
        if req_headers_user_agent_groups:
            df.loc[idx, 'req_headers_user_agent'] = ' '.join(req_headers_user_agent_groups.groups())
        if '443' in str(row.req_headers_x_forwarded_port):
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'HTTPS'
        elif '80' in str(row.req_headers_x_forwarded_port):
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'HTTP'
        elif '21' in str(row.req_headers_x_forwarded_port):
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'FTP'
        elif '22' in str(row.req_headers_x_forwarded_port):
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'SSH'
        elif '25' in str(row.req_headers_x_forwarded_port):
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'SMTP'
        elif '53' in str(row.req_headers_x_forwarded_port):
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'DNS'
        elif '8080' in str(row.req_headers_x_forwarded_port):
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'HTTP'
        else:
            df.loc[idx, 'req_headers_x_forwarded_port'] = 'UNKNOWN'

        # change timestamp to datetime format
        timestamp_groups = re.match(r'([0-9].*-[0-9].*-[0-9].*)T([0-9].*:[0-9].*:[0-9].*)Z', str(row.timestamp))
        if timestamp_groups:
            df.loc[idx, 'timestamp'] = ' '.join(timestamp_groups.groups())

        # if there is no user, set it to '-'
        if pd.isnull(row.req_headers_netguard_proxy_user):
            df.loc[idx, 'req_headers_netguard_proxy_user'] = '-'

    np.savetxt(F'{LOGS_CSV_OUTPUT_DIR}{UNPARSED_FILE}', df.values, fmt="%s")
