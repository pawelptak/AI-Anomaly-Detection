import torch
from sklearn.metrics import f1_score
import numpy as np
from plots import *


def compute_accuracy(model, data_loader, DEVICE):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE, dtype=torch.long)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_f1(model, data_loader, device):
    y_hats = []
    y_acts = []
    for i, (inputs, targets) in enumerate(data_loader):
        yhat = model(inputs)[-1].cpu().detach().numpy().round()
        yhat = np.argmax(yhat, axis=1)
        y_hats.append(yhat)
        y_acts.append(list(targets.cpu().detach().numpy()))

    y_hats = np.array([item for sublist in y_hats for item in sublist])
    y_acts = np.array([item for sublist in y_acts for item in sublist])
    return f1_score(y_acts, y_hats)