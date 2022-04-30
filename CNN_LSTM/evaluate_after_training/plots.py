from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
from settings import *


def display_confusion_matrix(target_names, Y_test, Y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(Y_test, Y_pred)
    title = 'Confusion matrix'

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.1f}; misclass={:0.1f}'.format(accuracy, misclass))
    plt.show()

def plot_cost_functions(minibatch_cost_cpu):
    plt.plot(range(len(minibatch_cost_cpu)), minibatch_cost_cpu)
    plt.ylabel('Cost Function Label')
    plt.xlabel('Minibatch')
    plt.show()


def plot_f1_score(epoch_train_performance, epoch_val_performance):
    plt.plot(range(len(epoch_train_performance)), epoch_train_performance, label="train f1 scores")
    plt.plot(range(len(epoch_val_performance)), epoch_val_performance, label="val f1 scores")
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()