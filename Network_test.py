import numpy as np
import torch

from Encoder import Encoder
import torch.optim as optim
import redis
import _pickle as cPickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bsize = 25
time_step = 10
obs = np.load("test.npy")
obs = np.repeat(obs[np.newaxis, ...], time_step, 0)
obs = np.repeat(obs[np.newaxis, ...], bsize, 0)
obs = torch.tensor(obs).float().to(device)
act = torch.zeros((bsize, time_step-1, 2)).float().to(device)
network = Encoder(cnn_out_size=100, action_out_size=16, lstm_hidden_size=15, action_shape=(2, 3), atten_size=7)
connect=redis.Redis("localhost")
connect.rpush("netwrok", cPickle.dumps(network.state_dict()))
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
# hidden_state, cell_state, lstm_out = network.lstm.init_hidden_states_and_outputs(bsize)
# lstm_out, (hidden_state, cell_state), dqn_out, out_per_action, (
# hidden_state_per_action, cell_state_per_action) = network(obs, act, bsize, hidden_state, cell_state, lstm_out)
print("Model's state_dict:")
for param_tensor in network.state_dict():
    print(param_tensor, "\t", network.state_dict()[param_tensor].size())

print()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])