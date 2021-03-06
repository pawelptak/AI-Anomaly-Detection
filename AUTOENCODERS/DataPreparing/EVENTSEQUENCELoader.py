import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import exists

"""
Class for Loading Data from EVENTSEQUENCE folder
The data was generated by running drain on a system logs file
"""


class EVENTSEQUENCELoader:
    def __init__(self):
        self.dataset_name = "EVENTSEQUENCE"
        self.data_basepath = f"Data/{self.dataset_name}"
        self.data_fullpath = f"{self.data_basepath}/all_events.csv"
        self.data_testpath = f"{self.data_basepath}/TEST.csv"
        self.data_trainpath = f"{self.data_basepath}/TRAIN.csv"

        self.label_column = 'Label'
        self.normal_label = 'Normal'
        self.samples = 10000

        self.normal_record = "['E22', 'E5', 'E5', 'E5', 'E26', 'E26', 'E26', 'E11', 'E9', 'E11', 'E9', 'E11', 'E9']"

        if(not(exists(self.data_trainpath)) or not(exists(self.data_testpath))):
            df = pd.read_csv(self.data_fullpath, header=0)

            df = df.drop(columns=["BlockId"])
            df.loc[df["EventSequence"] ==
                   self.normal_record, self.label_column] = self.normal_label

            train_part = df[:6300]
            train = train_part[train_part[self.label_column]
                               == self.normal_label]
            test = df[6300:]
            test = pd.concat([train_part[train_part[self.label_column]
                                         != self.normal_label], test])

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
