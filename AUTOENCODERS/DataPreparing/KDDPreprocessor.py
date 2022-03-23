from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tqdm


class KDDPreprocessor:
    def __init__(self, group_records=False, is_window=False, window_size=1, stride=1):
        self.categorical_columns = ['protocol_type', 'service', 'flag']
        self.group_records = group_records
        self.is_windows = is_window
        self.window_size = window_size
        self.stride = stride

        # 0 -> wpis normalny - Nie Atak
        # 1 -> DoS (Denial of Service)
        # 2 -> Probe
        # 3 -> R2L (Remote To User)
        # 4 -> U2R  (User To Root)
        self.attacks_categories = {'normal': 0,
                                   'neptune': 1, 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1, 'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                                   'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
                                   'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3, 'sendmail': 3, 'named': 3, 'snmpgetattack': 3, 'snmpguess': 3, 'xlock': 3, 'xsnoop': 3, 'httptunnel': 3,
                                   'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4, 'xterm': 4}

    def preprocess_train_data(self, df: pd.DataFrame, label='normal'):
        if(self.group_records):
            df['label'] = df['label'].replace(self.attacks_categories)

        df = df[df['label'] == label]
        df.reset_index(drop=True, inplace=True)
        self.train_categorical_unique = {col: np.array(sorted(
            df[col].unique())) for col in self.categorical_columns}

        self.onehotencoders = {col: OneHotEncoder(
            handle_unknown='ignore') for col in self.categorical_columns}

        for (col, encoder) in self.onehotencoders.items():
            encoder.fit(self.train_categorical_unique[col].reshape(-1, 1))

            encoded_columns = [
                f'{col}_{c}' for c in self.train_categorical_unique[col]]
            new_data = encoder.transform(
                df[col].values.reshape(-1, 1)).toarray()
            new_data_frame = pd.DataFrame(new_data, columns=encoded_columns)
            df = df.join(new_data_frame)
            df.drop(col, axis=1, inplace=True)

        df.drop('label', axis=1, inplace=True)
        self.scaler = MinMaxScaler()
        x = df.values  # returns a numpy array
        df = pd.DataFrame(self.scaler.fit_transform(x), columns=df.columns)

        if(self.is_windows):

            windows = self.__get_windows(df, self.window_size, self.stride)
            return windows

        return df

    def preprocess_test_data(self, df: pd.DataFrame, label='neptune'):
        if(self.group_records):
            df['label'] = df['label'].replace(self.attacks_categories)

        df = df[df['label'] == label]
        df.reset_index(drop=True, inplace=True)
        for (col, encoder) in self.onehotencoders.items():
            encoded_columns = [
                f'{col}_{c}' for c in self.train_categorical_unique[col]]
            new_data = encoder.transform(
                df[col].values.reshape(-1, 1)).toarray()
            new_data_frame = pd.DataFrame(new_data, columns=encoded_columns)
            df = df.join(new_data_frame)
            df.drop(col, axis=1, inplace=True)

        status = pd.Series(
            [0 if i == label else 1 for i in df['label']])

        df.drop('label', axis=1, inplace=True)
        x = df.values  # returns a numpy array
        df = pd.DataFrame(self.scaler.transform(x), columns=df.columns)

        if(self.is_windows):
            windows = self.__get_windows(df, self.window_size, self.stride)
            y_label = [1 if np.sum(status[i:i+self.window_size]) > 0 else 0 for i in range(
                0, len(df.values)-self.window_size+1, self.stride)]

            return(windows, y_label)

        return df

    def preprocess_test_data_multilabel(self, df: pd.DataFrame, normal_label, attack_label):
        if(self.group_records):
            df['label'] = df['label'].replace(self.attacks_categories)

        df = df[(df['label'] == normal_label) | (df['label'] == attack_label)]
        df.reset_index(drop=True, inplace=True)
        # df.to_csv(f'Results/test0_{label}.csv')
        for (col, encoder) in self.onehotencoders.items():
            encoded_columns = [
                f'{col}_{c}' for c in self.train_categorical_unique[col]]
            new_data = encoder.transform(
                df[col].values.reshape(-1, 1)).toarray()
            new_data_frame = pd.DataFrame(new_data, columns=encoded_columns)
            df = df.join(new_data_frame)
            df.drop(col, axis=1, inplace=True)

        # df.to_csv(f'Results/test1_{label}.csv')

        status = pd.Series(
            [0 if i == normal_label else 1 for i in df['label']])
        df.drop('label', axis=1, inplace=True)
        x = df.values  # returns a numpy array
        df = pd.DataFrame(self.scaler.transform(x), columns=df.columns)
        # df.to_csv(f'Results/test2_{label}.csv')

        if(self.is_windows):
            windows = self.__get_windows(df, self.window_size, self.stride)
            y_label = [1 if np.sum(status[i:i+self.window_size]) > 0 else 0 for i in range(
                0, len(df.values)-self.window_size+1, self.stride)]

            return(windows, y_label)

        return df

    def __get_windows(self, df, window_size=20, stride=10):
        windows_arr = []
        for i in tqdm.tqdm(range(0, len(df)-window_size+1, stride)):
            windows_arr.append(df.iloc[i:i+window_size, :].to_numpy())
        return np.array(windows_arr)
