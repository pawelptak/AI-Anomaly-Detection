import torch
import torch.nn as nn
from torch.optim import Adam
from evaluation.plots import *
from settings import *
from alive_progress import alive_bar


class logCNN(nn.Module):
    def __init__(self, config):
        super(logCNN, self).__init__()
        self.num_classes = config.num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=2),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(32 * 7 * 7, 10)

        self.lstm = nn.LSTM(config.lstm_input_size, config.hidden_size, config.num_layers, batch_first=True)
        self.hidden = self.init_hidden(config.batch_size, config.hidden_size, config.num_lstm_directions, config.num_layers)
        self.linear2 = nn.Linear(config.hidden_size, config.num_classes)

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

def train_model(model, train_loader, device, config):
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(config.num_epochs):
        with alive_bar(len(train_loader)-1, bar='circles', title=f"Epoch {epoch}") as bar:
            for i, (data, labels) in enumerate(train_loader):
                if data.shape[0] != config.batch_size:
                    break
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    bar.text(f"Epoch: {epoch}/{config.num_epochs}, Step: {i}, Loss: {loss.item()}")
                _, predicted = torch.max(outputs.data, 1)
                bar()
                
    return model

