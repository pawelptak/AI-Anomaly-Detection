import torch
import torch.nn as nn
from torch.optim import Adam
from evaluation.plots import *
from settings import *


class logCNN(nn.Module):
    def __init__(self, num_classes):
        super(logCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=2),
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
        return y


def train_model(model, train_loader, device):
    print("Start training...")
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(NUM_EPOCHS):
        for i, (data, labels) in enumerate(train_loader):
            if data.shape[0] != BATCH_SIZE:
                break
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("Epoch: {}/{}".format(epoch, NUM_EPOCHS),
                      "Step: {}".format(i),
                      "Loss: {}".format(loss.item()))
            _, predicted = torch.max(outputs.data, 1)
    return model
