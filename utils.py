import torch
import numpy as np
import pickle
import torch


# dqn_out(bsize, act_shape[0]...)
def find_optimal_action(dqn_out):
    dqn_out_clone = torch.clone(dqn_out)
    dqn_out_clone = np.array(dqn_out_clone.detach().cpu())
    act = np.zeros((dqn_out_clone.shape[0], 1, len(dqn_out_clone.shape) - 1))
    for batch in range(len(dqn_out_clone)):
        index = np.unravel_index(dqn_out_clone[batch].argmax(), dqn_out_clone[batch].shape)
        act[batch, 0] = np.array(index)
    return act

def convert_to_array(object):
    if torch.is_tensor(object):
        object=np.array(object.detach().cpu())
    return object
def save_obj(obj, name):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
