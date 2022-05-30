import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
This class is used to analyze the experiment results, plotting the results
"""


class AutoencoderResultsAnlyzer:
    def __init__(self):
        pass

    def feed(self, errors, labels, threshold, f_score, model_name, title):
        self.errors = errors
        self.labels = labels
        self.threshold = threshold
        self.f_score = f_score
        self.model_name = model_name
        self.title = title

    def plot_results(self):
        path = f'Results/fig/{self.model_name}/results.png'
        self.ensure_dir(path)

        normal_mask = self.labels == 0
        anomaly_mask = self.labels == 1
        error_normal = self.errors[normal_mask]
        error_anomaly = self.errors[anomaly_mask]

        x_plot_normal = np.array(range(len(self.labels)))[normal_mask]
        x_plot_anomaly = np.array(range(len(self.labels)))[anomaly_mask]

        plt.scatter(x_plot_normal, error_normal, c='b')
        plt.scatter(x_plot_anomaly, error_anomaly, c='r')
        if(self.threshold > 0):
            plt.axhline(y=self.threshold, c='y')
        plt.title(self.title)
        plt.savefig(path)

    def plot_confusion_matrix(self):
        path = f'Results/fig/{self.model_name}/confusion_matrix.png'
        self.ensure_dir(path)
        predicted = self.errors < self.threshold
        matrix = confusion_matrix(self.labels, predicted)
        plt.figure(figsize=(10, 8))
        colors = ["orange", "green"]
        sns.heatmap(matrix, xticklabels=["Anomaly", "Normal"],
                    yticklabels=["Normal", "Anomaly"], cmap=colors, annot=True, fmt="d")
        plt.title(f'Confusion Matrix, F-score = {self.f_score}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(path)

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
