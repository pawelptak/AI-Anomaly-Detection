import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from plots import *


import time

from settings import *
from metrics import compute_f1


class logCNN(nn.Module):
    def __init__(self, num_classes):
        super(logCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(32 * 7 * 7, 10)

        self.lstm = nn.LSTM(LSTM_INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.hidden = self.init_hidden(BATCH_SIZE, HIDDEN_SIZE, NUM_LSTM_DIRECTIONS, NUM_LAYERS)
        self.linear2 = nn.Linear(HIDDEN_SIZE, num_classes)

    def init_hidden(self, batch_size, hidden_size, num_directions, num_layers):
        return (torch.autograd.Variable(torch.zeros(num_directions * num_layers, batch_size, hidden_size)),
                torch.autograd.Variable(torch.zeros(num_directions * num_layers, batch_size, hidden_size)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x, self.hidden)
        y = self.linear2(x[:, -1, :])
        prob = F.softmax(y)
        return y, prob


def train_model(DEVICE, train_loader, val_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()
    minibatch_cost = []
    epoch_train_performance = []
    epoch_val_performance = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE, dtype=torch.long)  # another had to use torch.long

            ### FORWARD AND BACK PROP
            y, probas = model(features)
            cost = F.cross_entropy(y, targets)
            optimizer.zero_grad()
            cost.backward()
            minibatch_cost.append(cost)
            optimizer.step()

            if not batch_idx % 100:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                      % (epoch + 1, NUM_EPOCHS, batch_idx,
                         len(train_loader), cost,))

        model.eval()
        with torch.set_grad_enabled(False):  # save memory during inference
            train_performance = compute_f1(model, train_loader, device=DEVICE)
            val_performance = compute_f1(model, val_loader, device=DEVICE)
            epoch_train_performance.append(train_performance)
            epoch_val_performance.append(val_performance)
            print('Epoch: %03d/%03d | Train: %.3f%%   | Val: %.3f%%' % (
                epoch + 1, NUM_EPOCHS, train_performance, val_performance))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))


def evaluate_model(model, test_loader, train_loader):
    # Evaluate metrics on the test set
    y_hats = []
    y_acts = []
    counter = 0
    for i, (inputs, targets) in enumerate(test_loader):
        yhat = model(inputs)[-1].cpu().detach().numpy().round()
        yhat = np.argmax(yhat, axis=1)
        y_hats.append(yhat)
        y_acts.append(list(targets.cpu().detach().numpy()))
        counter += 1

    y_hats = [item for sublist in y_hats for item in sublist]
    y_acts = [item for sublist in y_acts for item in sublist]

    target_names = list(set(y_acts))
    display_confusion_matrix(target_names, y_acts, y_hats)


    print("TEST SET METRICS:")
    f1 = f1_score(y_acts, y_hats)
    print("f1 score : ", f1)
    precision = precision_score(y_acts, y_hats)
    print("precision", precision)
    recall = recall_score(y_acts, y_hats)
    print("recall", recall)

    test_ys = pd.DataFrame(list(zip(y_acts, y_hats)), columns=["y_true", "y_pred"])

    print("TEST SET:\n")
    print("anomalous:\n")
    test_anomalous = test_ys[test_ys["y_true"] == 1]
    print("number of anomalies in the test set:", len(test_anomalous))
    correct_anomalous = test_anomalous[test_anomalous["y_true"] == test_anomalous["y_pred"]]
    print("number of anomalies correctly identified", len(correct_anomalous))
    incorrect_anomalous = test_anomalous[test_anomalous["y_true"] != test_anomalous["y_pred"]]
    print("number of anomalies incorrectly identified", len(incorrect_anomalous))

    print("\nnormal:\n")
    test_normals = test_ys[test_ys["y_true"] == 0]
    print("number of normals in the test set:", len(test_normals))
    correct_normal = test_normals[test_normals["y_true"] == test_normals["y_pred"]]
    print("number of normals correctly identified", len(correct_normal))
    incorrect_normal = test_normals[test_normals["y_true"] != test_normals["y_pred"]]
    print("number of normals incorrectly identified", len(incorrect_normal))

    # Evaluate metrics on train set
    y_hats = []
    y_acts = []
    counter = 0
    for i, (inputs, targets) in enumerate(train_loader):
        yhat = model(inputs)[-1].cpu().detach().numpy().round()
        yhat = np.argmax(yhat, axis=1)
        y_hats.append(yhat)
        y_acts.append(list(targets.cpu().detach().numpy()))
        counter += 1

    y_hats = [item for sublist in y_hats for item in sublist]
    y_acts = [item for sublist in y_acts for item in sublist]

    print("TRAIN SET METRICS:")
    f1 = f1_score(y_acts, y_hats)
    print("f1 score : ", f1)
    precision = precision_score(y_acts, y_hats)
    print("precision", precision)
    recall = recall_score(y_acts, y_hats)
    print("recall", recall)

    train_ys = pd.DataFrame(list(zip(y_acts, y_hats)), columns=["y_true", "y_pred"])
    print("TRAIN SET:\n")
    print("anomalous:\n")
    train_anomalous = train_ys[train_ys["y_true"] == 1]
    print("number of anomalies in the train set:", len(train_anomalous))
    correct_anomalous = train_anomalous[train_anomalous["y_true"] == train_anomalous["y_pred"]]
    print("number of anomalies correctly identified", len(correct_anomalous))
    incorrect_anomalous = train_anomalous[train_anomalous["y_true"] != train_anomalous["y_pred"]]
    print("number of anomalies incorrectly identified", len(incorrect_anomalous))

    print("\nnormal:\n")
    train_normals = train_ys[train_ys["y_true"] == 0]
    print("number of normals in the train set:", len(train_normals))
    correct_normal = train_normals[train_normals["y_true"] == train_normals["y_pred"]]
    print("number of normals correctly identified", len(correct_normal))
    incorrect_normal = train_normals[train_normals["y_true"] != train_normals["y_pred"]]
    print("number of normals incorrectly identified", len(incorrect_normal))

    # Save the model
    torch.save(model.state_dict(), "model.pt")
