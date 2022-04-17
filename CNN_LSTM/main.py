import torch
from sklearn.model_selection import train_test_split

from sliding_window_processor import prepare_dataframe, FeatureExtractor
from prepare_dataset import prepare_custom_datasets, add_padding
from prepare_model import *
from logs_parser import *
from malicious_urls import *
from plots import *
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

PREPARE_DATA = True
if __name__ == '__main__':

    if PREPARE_DATA:
        parser.parse(log_file_all)
        validate_links_in_csv()
        logs_parsed = pd.read_csv("{}nsmc-kibana_new.txt_structured.csv".format(LOGS_PARSED_OUTPUT_DIR))
        x = logs_parsed.iloc[:]
        y = logs_parsed['label']
        re_pat = r"host=(\[.*])"

        # collecting events
        dataframe = prepare_dataframe(x, re_pat)
        dataframe.to_csv("{}events.csv".format(LOGS_PARSED_OUTPUT_DIR), index=False)

        # # merging block frames with labels
        # events_train = events_train.merge(y, on="BlockId")
        # events_test = events_test.merge(y, on="BlockId")

        # removing blocks that are overlapped into train and test
        # overlapping_blocks = np.intersect1d(events_train["BlockId"], events_test["BlockId"])
        # events_train = events_train[~events_train["BlockId"].isin(overlapping_blocks)]
        # events_test = events_test[~events_test["BlockId"].isin(overlapping_blocks)]

    #     events_train_values = events_train["EventSequence"].values
    #     events_test_values = events_test["EventSequence"].values
    #
    #     # fit transform
    #     fe = FeatureExtractor()
    #     subblocks_train = fe.fit_transform(
    #         events_train_values,
    #         term_weighting="tf-idf",
    #         length_percentile=95,
    #         window_size=16,
    #     )
    #     subblocks_test = fe.transform(events_test_values)
    #
    #     print("collecting y data")
    #     y_train = events_train[["BlockId", "Label"]]
    #     y_test = events_test[["BlockId", "Label"]]
    #
    #     # saving files
    #     print("writing y to csv")
    #     y_train.to_csv(f"{LOGS_PREPARED_OUTPUT_DIR}y_train.csv")
    #     y_test.to_csv(f"{LOGS_PREPARED_OUTPUT_DIR}y_test.csv")
    #
    #     print("saving x to numpy object")
    #     np.save(f"{LOGS_PREPARED_OUTPUT_DIR}x_train.npy", subblocks_train)
    #     np.save(f"{LOGS_PREPARED_OUTPUT_DIR}x_test.npy", subblocks_test)
    #
    #     train_data = subblocks_train
    #     read_train_labels = y_train
    #     train_labels = read_train_labels['Label'] == 'Anomaly'
    #     train_labels = train_labels.astype(int)
    #
    #     test_data = subblocks_test
    #     read_test_labels = y_test
    #     test_labels = read_test_labels['Label'] == 'Anomaly'
    #     test_labels = test_labels.astype(int)
    # else:
    #
    #     train_data = np.load(f"{LOGS_PREPARED_OUTPUT_DIR}x_train.npy")
    #     read_train_labels = pd.read_csv(f"{LOGS_PREPARED_OUTPUT_DIR}y_train.csv")
    #     train_labels = read_train_labels['Label'] == 'Anomaly'
    #     train_labels = train_labels.astype(int)
    #
    #     test_data = np.load(f"{LOGS_PREPARED_OUTPUT_DIR}x_test.npy")
    #     read_test_labels = pd.read_csv(f"{LOGS_PREPARED_OUTPUT_DIR}y_test.csv")
    #     test_labels = read_test_labels['Label'] == 'Anomaly'
    #     test_labels = test_labels.astype(int)
    #
    # train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
    #                                                                   random_state=42)
    # train_labels = train_labels.reset_index(drop=True)
    # val_labels = val_labels.reset_index(drop=True)
    #
    # train_data, test_data, val_data = add_padding(train_data, test_data, val_data)
    # train_loader, test_loader, val_loader = prepare_custom_datasets(train_data, test_data, val_data, train_labels, test_labels, val_labels)
    #
    # model = logCNN(NUM_CLASSES)
    # model.to(DEVICE)
    #
    # # training
    # epoch_train_performance, epoch_val_performance, minibatch_cost = train_model(DEVICE, train_loader,
    #                                                                              val_loader, model)
    #
    # # plots
    # minibatch_cost_cpu = [i.cpu().detach().numpy() for i in minibatch_cost]
    # plot_cost_functions(minibatch_cost_cpu)
    # plot_f1_score(epoch_train_performance, epoch_val_performance)
    #
    # evaluate_model(model, test_loader, train_loader)
