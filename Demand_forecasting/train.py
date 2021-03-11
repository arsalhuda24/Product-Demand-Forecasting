import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
from torch.autograd import Variable
from model import LSTM
from loader import get_loader
from sklearn.preprocessing import MinMaxScaler
import os


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = 1

    path = "C:\\Users\\arsal\\Downloads\\Demand_forecasting\\data"

    loader = get_loader(path)

    model = LSTM(lstm_input_size, h1, batch_size=25, output_dim=output_dim, num_layers=num_layers)

    model.apply(init_weights)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for _, batch in enumerate(loader):
            """..."""



input_size = 20
# If `per_element` is True, then LSTM reads in one timestep at a time.
per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
# size of hidden layers
h1 = 32
output_dim = 1
num_layers = 2
learning_rate = 1e-3
num_epochs = 500


model = LSTM(lstm_input_size, h1, batch_size=25, output_dim=output_dim, num_layers=num_layers)
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

""""Train Model"""

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()



