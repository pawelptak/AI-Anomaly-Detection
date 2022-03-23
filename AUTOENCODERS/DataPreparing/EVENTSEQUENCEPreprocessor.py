from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tqdm
import ast


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


class EVENTSEQUENCEPreprocessor:
    def __init__(self):
        self.label_column = 'Label'
        self.normal_label = 'Normal'

    def preprocess_train_data(self, df: pd.DataFrame):
        df.drop([0], inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.max_length = 0
        for index, value in enumerate(df.values):
            sequence = value[0]
            sequence_array = np.array(ast.literal_eval(sequence))
            self.max_length = max(self.max_length, len(sequence_array))

        self.max_length += 3
        unique_sequences = df["EventSequence"].unique()
        self.unique_event = np.unique(
            flat_map(lambda s: ast.literal_eval(s), unique_sequences))

        self.onehotencoder = OneHotEncoder(handle_unknown='ignore')
        self.onehotencoder.fit(self.unique_event.reshape(-1, 1))

        data = np.zeros(
            (len(df.values), self.max_length, len(self.unique_event)))

        for index, value in enumerate(df.values):
            sequence = value[0]
            sequence_array = np.array(ast.literal_eval(sequence))
            transformed = self.onehotencoder.transform(
                sequence_array.reshape(-1, 1)).toarray()
            data[index, :len(sequence_array)] = transformed

        return data

    def preprocess_test_data(self, df: pd.DataFrame):
        df.drop([0], inplace=True)

        y_label = pd.Series(
            [0 if i == self.normal_label else 1 for i in df[self.label_column]])

        data = np.zeros(
            (len(df.values), self.max_length, len(self.unique_event)))

        for index, value in enumerate(df.values):
            sequence = value[0]
            sequence_array = np.array(ast.literal_eval(sequence))
            sequence_array = sequence_array[:self.max_length]
            transformed = self.onehotencoder.transform(
                sequence_array.reshape(-1, 1)).toarray()
            data[index, :len(sequence_array)] = transformed

        return(data, np.array(y_label))
