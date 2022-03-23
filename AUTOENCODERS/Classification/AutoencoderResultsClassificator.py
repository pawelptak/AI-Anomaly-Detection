import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf


class AutoencoderResultsClassificator:
    def __init__(self):
        pass

    def feed(self, X_0, Y_0, X_1, Y_1):
        self.X_0 = X_0
        self.X_1 = X_1
        self.Y_0 = Y_0
        self.Y_1 = Y_1

    def calculate_reconstruction_error(self):
        errors_0 = self.__calculate_reconstruction_error(self.X_0, self.Y_0)
        errors_1 = self.__calculate_reconstruction_error(self.X_1, self.Y_1)

        # skalowanie bledow rekonstrukcji do zakresu [0, 1]
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(np.concatenate((errors_0, errors_1)).reshape(-1, 1))

        self.error_0 = mm_scaler.transform(errors_0.reshape(-1, 1))
        self.error_1 = mm_scaler.transform(errors_1.reshape(-1, 1))

        return (self.error_0, self.error_1)

    def calculate_reconstruction_error_windows(self):
        errors_0 = self.__calculate_reconstruction_error_windows(
            self.X_0, self.Y_0)
        errors_1 = self.__calculate_reconstruction_error_windows(
            self.X_1, self.Y_1)

        # skalowanie bledow rekonstrukcji do zakresu [0, 1]
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(np.concatenate((errors_0, errors_1)).reshape(-1, 1))

        self.error_0 = mm_scaler.transform(errors_0.reshape(-1, 1))
        self.error_1 = mm_scaler.transform(errors_1.reshape(-1, 1))

        return (self.error_0, self.error_1)

    def calculate_best_threshold(self) -> float:
        self.labels = self.__get_labels(self.X_0, self.X_1)
        self.scores = np.concatenate((self.error_0, self.error_1))

        fpr, tpr, thresholds = roc_curve(self.labels, self.scores)

        auc = roc_auc_score(self.labels, self.scores)

        # self.__plot_roc(fpr, tpr, auc, "TEST")
        anomaly_combinations = [(self.scores > i).astype(np.int32)
                                for i in thresholds]
        f1_scores = [f1_score(self.labels, i) for i in anomaly_combinations]
        self.max_f1_score = np.max(f1_scores)
        self.best_threshold = thresholds[f1_scores.index(self.max_f1_score)]

        #self.__plot_threshold_curve(thresholds, f1_scores, "TEST")

        return (self.max_f1_score, self.best_threshold)

    def classify(self, X):
        pass

    def __calculate_reconstruction_error(self, X, Y):
        return np.sum(np.abs(np.subtract(X, Y)), axis=1)

    def __calculate_reconstruction_error_windows(self, true_windows, pred_windows):
        reconstruction_errors = []
        window_size = true_windows.shape[1]
        features_number = true_windows.shape[2]

        def __cond(y_true, y_pred, i, iters):
            return tf.less(i, iters)

        def __body(y_true, y_pred, i, iters):
            tensor_for_error = tf.math.subtract(tf.slice(
                y_true, [i, 0, 0], [1, -1, -1]), tf.slice(y_pred, [i, 0, 0], [1, -1, -1]))
            tensor_for_error = tf.reshape(
                tensor_for_error, [window_size, features_number])
            reconstruction_error = tf.math.reduce_mean(
                tf.norm(tensor_for_error, ord='euclidean', axis=1))
            reconstruction_errors.append(reconstruction_error.numpy())
            return [y_true, y_pred, tf.add(i, 1), iters]

        iters = tf.constant(len(true_windows))

        result = tf.while_loop(__cond, __body, [tf.constant(true_windows.astype(
            np.float32)), tf.constant(pred_windows.astype(np.float32)), 0, iters])
        return reconstruction_errors

    def __get_labels(self, X_0, X_1):
        return np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

    def __plot_roc(self, fpr, tpr, auc, model_name):
        plt.figure(figsize=(10, 5))
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.plot(fpr, tpr, label='AUC={}'.format(auc))
        plt.grid()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('ROC')
        plt.show()

    def __plot_threshold_curve(self, thresholds, f1_scores, model_name):
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, f1_scores)
        plt.grid()
        plt.xlabel('Thresholds')
        plt.ylabel('F-1 Score')
        plt.title('F-1 Score vs Thresholds')
        plt.show()
