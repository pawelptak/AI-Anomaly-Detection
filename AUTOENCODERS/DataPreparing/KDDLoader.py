import pandas as pd


class KDDLoader:
    def __init__(self, service="http"):
        self.dataset_name = "KDD"
        self.data_basepath = f"Data/{self.dataset_name}"
        self.data_trainpath = f"{self.data_basepath}/TRAIN.csv"
        self.data_testpath = f"{self.data_basepath}/TEST.csv"
        self.service = service

        self.col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                          "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                          "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                          "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                          "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                          "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                          "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                          "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                          "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                          "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    """ load train data """

    def load_train_data(self):
        return self.__load_data(self.data_trainpath)

    """ load test data """

    def load_test_data(self):
        return self.__load_data(self.data_testpath)

    """ load additional data for further predictions (optional) """

    def load_predict_data(self, path: str):
        return self.__load_data(path)

    def __load_data(self, path: str):
        df = pd.read_csv(path, header=None, names=self.col_names)

        if(self.service != ""):
            df = df[df["service"] == self.service]

        return df
