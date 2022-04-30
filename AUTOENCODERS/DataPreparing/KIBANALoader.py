import pandas as pd
import json


class KIBANALoader:
    def __init__(self):
        self.dataset_name = "KIBANA"
        self.data_basepath = f"Data/{self.dataset_name}"
        self.data_trainpath = f"{self.data_basepath}/nsmc-kibana-belk-kibana-5746d988d8-5f4lm_NORMAL.log"
        self.data_testpath = f"{self.data_basepath}/nsmc-kibana-belk-kibana-5746d988d8-5f4lm_ATTACK.log"

    def load_train_data(self):
        return self.__load_data(self.data_trainpath)

    def load_test_data(self):
        return self.__load_data(self.data_testpath)

    def load_test_data_lines(self):
        with open(self.data_testpath) as file:
            return file.readlines()

    def load_predict_data(self, path: str):
        return self.__load_data(path)

    def __load_data(self, path: str):
        with open(path) as json_file:
            lines = json_file.readlines()
            data = list(map(lambda l: json.loads(l), lines))

        df = pd.json_normalize(data)

        return df
