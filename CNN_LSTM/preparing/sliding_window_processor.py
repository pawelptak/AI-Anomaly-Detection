"""
loads and preprocesses the structured log data for anomaly prediction
"""
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from collections import Counter
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import math
from collections import Counter

def prepare_dataframe(data_frame, regex_pattern):
    """
    turns input data_frame into a 2 columned dataframe
    with columns: BlockId, EventSequence
    where EventSequence is a list of the events that happened to the block
    """
    data_list = []
    for _, row in data_frame.iterrows():
        blk_id_list = re.findall(regex_pattern, row["Content"])
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            data_list.append({"Source host": str(blk_id),
                              "EventId": row["EventId"],
                              "url_malicious_score": float(row["url_malicious_score"]),
                              "time [ms]": float(row["time [ms]"]),
                              "size [B]": float(row["size [B]"]),
                              "label": str(row["label"])
                              })

    data_frame = pd.DataFrame(data_list)
    return data_frame


def windower(sequence, window_size):
    """
    creates an array of arrays of windows
    output array is of length: len(sequence) - window_size + 1
    """
    return np.lib.stride_tricks.sliding_window_view(sequence, window_size)


def sequence_padder(sequence, required_length):
    """
    right pads events sequence until max sequence length long
    """
    if len(sequence) > required_length:
        return sequence
    return np.pad(
        sequence,
        (0, required_length - len(sequence)),
        mode="constant",
        constant_values=(0),
    )


def resize_time_image(time_image, size):
    """
    compresses time images that had more sequences then the set max sequence length
    """
    width = size[1]
    height = size[0]
    return np.array(Image.fromarray(time_image).resize((width, height)))


def fit_transform2(data_frame, max_seq_length, window_size):
    import math
    from collections import Counter
    all_events = data_frame[["EventId"]].values
    all_events = all_events.reshape(-1)
    all_events_dict = dict(Counter(all_events))
    matrix_size = (6, 4)
    columns_in_one_log = 4
    logs_in_one_matrix = math.floor((matrix_size[0] * matrix_size[1])/columns_in_one_log)
    number_of_all_matrixes = len(all_events) - 8
    data_frame["url_malicious_score"] = MinMaxScaler().fit_transform(data_frame["url_malicious_score"].values.reshape(-1, 1))
    data_frame["time [ms]"] = MinMaxScaler().fit_transform(data_frame["time [ms]"].values.reshape(-1, 1))
    data_frame["size [B]"] = MinMaxScaler().fit_transform(data_frame["size [B]"].values.reshape(-1, 1))
    rows_number = data_frame.shape[0]
    number_of_matrixes = rows_number - columns_in_one_log - 1
    labels = []
    x = np.zeros((number_of_matrixes, matrix_size[0], matrix_size[1]))
    for i in range(number_of_matrixes):
        window = data_frame.iloc[i:i+logs_in_one_matrix, :]
        window_labels = window.iloc[:, -1]
        features = window[["EventId", "url_malicious_score", "time [ms]", "size [B]"]]
        if features.shape[0] != 6:
            break
        events_in_window = dict(Counter(list(features.EventId.values)))
        for index, row in enumerate(features.itertuples()):
            features.iloc[index, 0] = math.log(number_of_all_matrixes/all_events_dict[row.EventId])
        features_numpy = np.array(features.values)
        if np.any(window_labels == "Malicious"):
            labels.append(1)
        else:
            labels.append(0)
        x[i] = features_numpy
    labels = np.array(labels)
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    print(x.shape)
    return x, labels


class FeatureExtractor(object):
    """
    class for fitting and transforming the training set
    then transforming the testing set
    """

    def __init__(self):
        self.mean_vec = None
        self.idf_vec = None
        self.events = None
        self.term_weighting = None
        self.max_seq_length = None
        self.window_size = None
        self.num_rows = None

    def fit_transform(
        self, X_seq, term_weighting=None, length_percentile=90, window_size=16
    ):
        """
        Fit and transform the training set
        X_Seq: ndarray,  log sequences matrix
        term_weighting: None or `tf-idf`
        length_percentile: int, set the max length of the event sequences
        window_size: int, size of subsetting
        """
        self.term_weighting = term_weighting
        self.window_size = window_size

        # get unique events
        self.events = set(np.concatenate(X_seq).ravel().flatten())

        # get lengths of event sequences
        length_list = np.array(list(map(len, X_seq)))
        self.max_seq_length = int(np.percentile(length_list, length_percentile))

        self.num_rows = self.max_seq_length - self.window_size + 1

        print("final shape will be ", self.num_rows, len(self.events))

        # loop over each sequence to create the time image
        time_images = []
        for block in X_seq:
            padded_block = sequence_padder(block, self.max_seq_length)
            time_image = windower(padded_block, self.window_size)
            time_image_counts = []
            for time_row in time_image:
                row_count = Counter(time_row)
                time_image_counts.append(row_count)

            time_image_df = pd.DataFrame(time_image_counts, columns=self.events)
            time_image_df = time_image_df.reindex(sorted(time_image_df.columns), axis=1)
            time_image_df = time_image_df.fillna(0)
            time_image_np = time_image_df.to_numpy()

            # resize if too large
            if len(time_image_np) > self.num_rows:
                time_image_np = resize_time_image(
                    time_image_np, (self.num_rows, len(self.events)),
                )

            time_images.append(time_image_np)

        # stack all the blocks
        X = np.stack(time_images)

        if self.term_weighting == "tf-idf":

            # set up sizing
            dim1, dim2, dim3 = X.shape
            X = X.reshape(-1, dim3)

            # apply tf-idf
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(dim1 / (df_vec + 1e-8))
            idf_tile = np.tile(self.idf_vec, (dim1 * dim2, 1))
            idf_matrix = X * idf_tile
            X = idf_matrix

            # reshape to original dimensions
            X = X.reshape(dim1, dim2, dim3)

        X_new = X
        print("train data shape: ", X_new.shape)
        return X_new

    def transform(self, X_seq):
        """
        transforms x test
        X_seq : log sequence data
        """

        # loop over each sequence to create the time image
        time_images = []
        for block in X_seq:
            padded_block = sequence_padder(block, self.max_seq_length)
            time_image = windower(padded_block, self.window_size)
            time_image_counts = []
            for time_row in time_image:
                row_count = Counter(time_row)
                time_image_counts.append(row_count)

            time_image_df = pd.DataFrame(time_image_counts, columns=self.events)
            time_image_df = time_image_df.reindex(sorted(time_image_df.columns), axis=1)
            time_image_df = time_image_df.fillna(0)
            time_image_np = time_image_df.to_numpy()

            # resize if too large
            if len(time_image_np) > self.num_rows:
                time_image_np = resize_time_image(
                    time_image_np, (self.num_rows, len(self.events)),
                )

            time_images.append(time_image_np)

        # stack all the blocks
        X = np.stack(time_images)

        if self.term_weighting == "tf-idf":

            # set up sizing
            dim1, dim2, dim3 = X.shape
            X = X.reshape(-1, dim3)

            # apply tf-idf
            idf_tile = np.tile(self.idf_vec, (dim1 * dim2, 1))
            idf_matrix = X * idf_tile
            X = idf_matrix

            # reshape to original dimensions
            X = X.reshape(dim1, dim2, dim3)

        X_new = X
        print("test data shape: ", X_new.shape)
        return X_new
