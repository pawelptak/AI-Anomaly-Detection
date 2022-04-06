from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tqdm


class KIBANAPreprocessor:
    def __init__(self):
        self.columns_to_keep = ['type', 'method',
                                'statusCode']  # , 'message', 'req.url']
        self.categorical_columns = ['type', 'method', 'statusCode']

    def preprocess_train_data(self, df: pd.DataFrame):
        df = df.filter(self.columns_to_keep)
        df = df.astype({col: str for col in self.categorical_columns})

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

        # self.scaler = MinMaxScaler()
        # x = df.values  # returns a numpy array
        # df = pd.DataFrame(self.scaler.fit_transform(x), columns=df.columns)

        return df

    def preprocess_test_data(self, df: pd.DataFrame):
        df = df.filter(self.columns_to_keep)
        df = df.astype({col: str for col in self.categorical_columns})

        for (col, encoder) in self.onehotencoders.items():
            encoded_columns = [
                f'{col}_{c}' for c in self.train_categorical_unique[col]]
            new_data = encoder.transform(
                df[col].values.reshape(-1, 1)).toarray()
            new_data_frame = pd.DataFrame(new_data, columns=encoded_columns)
            df = df.join(new_data_frame)
            df.drop(col, axis=1, inplace=True)

        return df
