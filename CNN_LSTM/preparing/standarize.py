# import pandas as pd
#
# def standarize_column(df, column):
#     df[column] = (df[column] - df[column].mean()) / df[column].std()
#
#
# def standarize_df(df_path, columns_to_standarize):
#     df = pd.read_csv(df_path)
#     for col in columns_to_standarize:
#         standarize_column(df, col)
#     df.to_csv(df_path)
#
#
# if __name__ == '__main__':
#     df_path = '../data/logs_parsed/events.csv'
#     standarize_df(df_path, columns_to_standarize=['url_malicious_score', 'time [ms]', 'size [B]'])