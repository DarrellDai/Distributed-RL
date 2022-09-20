import numpy as np
import torch

from Network import Network
from LSTM_attention import LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bsize = 25
time_step = 10
obs = np.load("test.npy")
obs = np.repeat(obs[np.newaxis, ...], time_step, 0)
obs = np.repeat(obs[np.newaxis, ...], bsize, 0)
obs = torch.tensor(obs).float().to(device)
act = torch.zeros((bsize, time_step, 2))
network = Network(cnn_out_size=100, lstm_hidden_size=15, action_shape=(2, 3), atten_size=7)
hidden_state, cell_state, lstm_out = network.lstm.init_hidden_states_and_outputs(bsize)
lstm_out, (hidden_state, cell_state), dqn_out, out_per_action, (
hidden_state_per_action, cell_state_per_action) = network.forward(obs, act, bsize, hidden_state, cell_state, lstm_out)
