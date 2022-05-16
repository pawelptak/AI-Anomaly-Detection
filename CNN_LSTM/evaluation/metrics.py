import torch
from sklearn.metrics import f1_score
import numpy as np
from evaluation.plots import *
from settings import BATCH_SIZE


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



def test_model(test_loader, train_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            if data.shape[0] != BATCH_SIZE:
                break
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            not_correct = (predicted != labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Not correct: %d' % not_correct)

    # test train data
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in train_loader:
            if data.shape[0] != BATCH_SIZE:
                break
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            not_correct = (predicted != labels).sum().item()
    print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
    print('Not correct: %d' % not_correct)