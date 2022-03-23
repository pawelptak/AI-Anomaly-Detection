import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import exists


class CICIDSLoaderWindows:
    def __init__(self, attack='DDoS'):
        self.dataset_name = "CICIDS2017"
        self.data_basepath = f"Data/{self.dataset_name}"
        self.data_fullpath = f"{self.data_basepath}/CICIDS2017_preprocessed.csv"
        self.data_testpath = f"{self.data_basepath}/CICIDS2017_WINDOWS_TEST.csv"
        self.data_trainpath = f"{self.data_basepath}/CICIDS2017_WINDOWS_TRAIN.csv"

        self.labal_column = ' Label'
        self.normal_label = 'BENIGN'
        self.attack_label = attack
        self.samples = 10000

        if(not(exists(self.data_trainpath)) or not(exists(self.data_testpath))):
            df = pd.read_csv(self.data_fullpath)
            df.rename(columns=lambda x: x.strip())
            # df = df[(df[self.labal_column] == self.normal_label) |
            #         (df[self.labal_column] == self.attack_label)]

            # print(df.columns)

            train = df[:100000]
            test = df[300000:400000]

            train.to_csv(self.data_trainpath, index=False)
            test.to_csv(self.data_testpath, index=False)

    def load_train_data(self):
        return self.__load_data(self.data_trainpath)

    def load_test_data(self):
        return self.__load_data(self.data_testpath)

    def __load_data(self, path: str):
        df = pd.read_csv(path)

        return df
