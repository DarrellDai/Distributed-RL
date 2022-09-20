import torch
import numpy as np
import pickle
import torch


# dqn_out(bsize, act_shape[0]...)
def find_optimal_action(dqn_out):
    dqn_out = np.array(dqn_out.detach())
    act = np.zeros((dqn_out.shape[0], 1, len(dqn_out.shape) - 1))
    for batch in range(len(dqn_out)):
        index = np.unravel_index(dqn_out[batch].argmax(), dqn_out[batch].shape)
        act[batch, 0] = np.array(index)
    return act


def save_obj(obj, name):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
