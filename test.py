import numpy as np
import torch

from Network import Network
from LSTM_attention import LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bsize = 25
time_step = 10
ob = np.load("test.npy")
ob = np.repeat(ob[np.newaxis, ...], time_step, 0)
ob = np.repeat(ob[np.newaxis, ...], bsize, 0)

network = Network(cnn_out_size=100, lstm_hidden_size=15, num_action=9)
hidden_state, cell_state, lstm_out = network.lstm.init_hidden_states_and_outputs(bsize)
dqn_out, (h_n, c_n) = network.forward(ob, bsize, time_step, hidden_state, cell_state, lstm_out)
