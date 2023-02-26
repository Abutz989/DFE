import numpy as np
import random
from typing import Any, Dict, Optional, Type, TypeVar, Union
import copy
from collections import namedtuple, deque
import sys
sys.path.append('..')
from  Algs.model import LSTM,TimeSeriesTransformer

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

BUFFER_SIZE = int(5e3)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4         # learning rate of the actor

WEIGHT_DECAY = 0.9   # L2 weight decay
LEARN_NUM = 10          # number of learning passes
LEARN_EVERY = BUFFER_SIZE        # learning timestep interval
damping_factor= 1e-5
epsilon = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size,hidden_size):
        super(LinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        return self.fc2(x)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, hidden_size, random_seed,env,batch_size = 128,model_type = 'MLP'):
        self.env = env
        self.model_type = model_type
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.epsilon = epsilon

        if model_type == 'LSTM' :
            self.model = LSTM(state_size=self.state_size, hidden_size=self.hidden_size, num_layers=2)
        elif model_type == 'Transformer' :
            self.model = TimeSeriesTransformer(input_size=self.state_size, output_size=self.state_size, num_layers=2,
                                               hidden_size=self.hidden_size, num_heads=8)
        else:
            self.model = LinearModel(input_size=self.state_size, output_size=self.state_size, hidden_size=self.hidden_size)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = LR)
        # obs = [Ph_err(3), pos_err(6), rotor_vel_err(4), cntr_cmd(4), Act(4)]
        column_names = ['Ph_err_' + str(i + 1) for i in range(3)] + \
                       ['pos_err_' + str(i + 1) for i in range(6)] + \
                       ['rotor_vel_err_' + str(i + 1) for i in range(4)] + \
                       ['cntr_cmd_' + str(i + 1) for i in range(4)] + \
                       ['Act_' + str(i + 1) for i in range(4)]
        # Create an empty DataFrame with the column names
        self.df = pd.DataFrame(data={}, columns=column_names)
        # Replay memory
        self.experience = []


    def step(self,max_t):

        # Save experience / reward
        # obs =  [Ph_err(3), pos_err(6),rotor_vel_err(4),cntr_cmd(4), Act(4)]
        self.env.reset()
        action = np.random.rand(4)*2
        for t in range(max_t):
            action =+ 3*np.ones(4)*np.sin(0.5*6.28*t)
            obs, r, d, i = self.env.step(action)  # send actions to environment
            self.df.loc[t] = obs

        self.learn()


    def test(self, max_t):

        # Save experience / reward
        # obs =  [Ph_err(3), pos_err(6),rotor_vel_err(4),cntr_cmd(4), Act(4)]
        self.env.reset()
        action = np.random.rand(4) * 2
        for t in range(max_t):
            action = + np.ones(4) * np.sin(0.2 * 6.28 * t)
            obs, r, d, i = self.env.step(action)  # send actions to environment
            self.df.loc[t] = obs

        score = self.eval()

        return score

    def learn(self):
        # df = self.df
        # for i in range(len(df) - 1):
        #     # Get the input and target data for the current row and the next row
        #     input_data = torch.tensor(df.iloc[i].values, dtype=torch.float32).unsqueeze(0)
        #     target_data = torch.tensor(df.iloc[i + 1].values, dtype=torch.float32).unsqueeze(0)
        #     # Zero the gradients of the model parameters
        #     self.optimizer.zero_grad()
        #     # Forward pass: compute the predicted output from the input
        #     predicted_output = self.model(input_data)
        #
        #     # Compute the loss between the predicted output and the target output
        #     loss = self.criterion(predicted_output, target_data)
        #
        #     # Backward pass: compute the gradients of the loss with respect to the model parameters
        #     loss.backward()
        #
        #     # Update the model parameters using the optimizer
        #     self.optimizer.step()


        # Convert the DataFrame to a PyTorch tensor
        data = torch.tensor(self.df.values, dtype=torch.float32)

        if self.model_type == 'Transformer':
            window_size = 11
            # Extract the number of features and time steps from the input tensor
            time_steps ,num_features = data.shape
            # Initialize an empty tensor to hold the rearranged data
            output_tensor = torch.zeros(time_steps - window_size, num_features, window_size)
            # Loop through the time steps and populate the output tensor
            for i in range(time_steps - window_size):
                output_tensor[i] = data[i:i + window_size,:].transpose(0, 1)
            data = output_tensor
        # Split the data into batches
        batches = torch.split(data, self.batch_size)
        # Train the linear model in batches
        for batch in batches:
            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute the predicted output for the current batch
            predicted_output = self.model(batch[:-1])
            # Compute the loss between the predicted output and the ground truth output
            if self.model_type == 'Transformer':
                loss = self.criterion(predicted_output, batch[1:,:,-1])
            else:
                loss = self.criterion(predicted_output, batch[1:])
            # Backpropagate the loss and update the model parameters
            loss.backward()
            # Update the model parameters using the optimizer
            self.optimizer.step()

    def eval(self):

        # test_data
        # with torch.no_grad():
        #     test_loss = 0.0
        #     for i, inputs in enumerate(test_data):
        #         outputs = self.model(inputs)
        #         loss = self.criterion(outputs, inputs[-1])
        #         test_loss += loss.item()
        #
        # return test_loss/len(test_data)

        df = self.df
        predicted_df = pd.DataFrame(columns=df.columns)
        if self.model_type == 'Transformer':
            window_size = 11
            for i in range(len(df)-window_size):
                # Get the input data for the current row
                input_data = torch.tensor(df.iloc[i:i + window_size].values, dtype=torch.float32).unsqueeze(0).transpose(-2,-1)
                # Forward pass: compute the predicted output from the input
                predicted_output = self.model(input_data)

                # Convert the predicted output back to a numpy array and add it as a new row to the predicted DataFrame
                predicted_row = pd.Series(predicted_output.detach().numpy()[0], index=df.columns)
                predicted_df = predicted_df.append(predicted_row, ignore_index=True)
        else:
            for i in range(len(df)):
                # Get the input data for the current row
                input_data = torch.tensor(df.iloc[i].values, dtype=torch.float32).unsqueeze(0)
                # Forward pass: compute the predicted output from the input
                predicted_output = self.model(input_data)

                # Convert the predicted output back to a numpy array and add it as a new row to the predicted DataFrame
                predicted_row = pd.Series(predicted_output.detach().numpy()[0], index=df.columns)
                predicted_df = predicted_df.append(predicted_row, ignore_index=True)
        # Evaluate the predicted DataFrame against the ground truth DataFrame
        mse = ((predicted_df - df.shift(-1)) ** 2).mean().mean()
        return mse
