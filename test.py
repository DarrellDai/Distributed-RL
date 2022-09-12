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
act=torch.zeros((bsize,time_step,2))
network = Network(cnn_out_size=100, lstm_hidden_size=15, action_shape=(2,3))
hidden_state, cell_state, lstm_out = network.lstm.init_hidden_states_and_outputs(bsize)
dqn_out, out_per_action, (h_n, c_n) = network.forward(ob, act, bsize, time_step, hidden_state, cell_state, lstm_out)
