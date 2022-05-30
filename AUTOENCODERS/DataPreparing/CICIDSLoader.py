import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import exists

"""
Class for Loading CICIDS2017 Data
The full data can be found here:
https://www.unb.ca/cic/datasets/ids-2017.html
"""


class CICIDSLoader:
    def __init__(self, attack):
        self.dataset_name = "CICIDS2017"
        self.data_basepath = f"Data/{self.dataset_name}"
        self.data_fullpath = f"{self.data_basepath}/CICIDS2017_preprocessed.csv"
        self.data_testpath = f"{self.data_basepath}/CICIDS2017_TEST_{attack}.csv"
        self.data_trainpath = f"{self.data_basepath}/CICIDS2017_TRAIN_{attack}.csv"

        self.labal_column = ' Label'
        self.normal_label = 'BENIGN'
        self.attack_label = attack
        self.samples = 10000

        if(not(exists(self.data_trainpath)) or not(exists(self.data_testpath))):
            df = pd.read_csv(self.data_fullpath)
            df.rename(columns=lambda x: x.strip())
            df = df[(df[self.labal_column] == self.normal_label) |
                    (df[self.labal_column] == self.attack_label)]

            print(df.columns)

            train = df[200000:300000]
            test = df[300000:304000]

            train.to_csv(self.data_trainpath, index=False)
            test.to_csv(self.data_testpath, index=False)

    def load_train_data(self):
        return self.__load_data(self.data_trainpath)

    def load_test_data(self):
        return self.__load_data(self.data_testpath)

    def load_predict_data(self, path: str):
        return self.__load_data(path)

    def __load_data(self, path: str):
        df = pd.read_csv(path)

        return df
