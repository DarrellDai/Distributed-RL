from Network import Network
import numpy as np
import pickle
import torch
import os
import re
import torch.nn as nn
import time
from mpi4py import MPI


def initialize_model(agent_ids, cnn_out_size, lstm_hidden_size, action_shape, action_out_size, atten_size):
    main_model = {}
    for idx in range(len(agent_ids)):
        main_model[agent_ids[idx]] = Network(cnn_out_size=cnn_out_size[idx],
                                             lstm_hidden_size=lstm_hidden_size[idx],
                                             atten_size=atten_size[idx], action_space_shape=tuple(action_shape[idx]),
                                             action_out_size=action_out_size[idx])
    return main_model

def wrap_model_with_dataparallel(models, device_idx):
    device = torch.device('cuda:' + str(device_idx[0]) if torch.cuda.is_available() else 'cpu')
    for id in models:
        if torch.cuda.is_available():
            models[id] = nn.DataParallel(models[id], device_ids=device_idx).to(device)
        else:
            models[id] = models[id].to(device)

def get_agents_id_to_name(env):
    agent_ids = tuple(env.agent_id_to_behaviour_name.keys())
    id_to_name = find_name_of_agents(env.agent_id_to_behaviour_name, agent_ids)
    return id_to_name

def find_name_of_agents(agent_id_to_behaviour_name, agent_ids):
    agent_id_to_name = {}
    hider_idx = 0
    seeker_idx = 0
    for id in agent_ids:
        if re.split("\?", agent_id_to_behaviour_name[id])[0] == "Hider":
            agent_id_to_name[id] = "Hider " + str(hider_idx)
            hider_idx += 1
        elif re.split("\?", agent_id_to_behaviour_name[id])[0] == "Seeker":
            agent_id_to_name[id] = "Seeker " + str(seeker_idx)
            seeker_idx += 1
    return agent_id_to_name


# dqn_out(bsize, act_shape[0]...)
def find_optimal_action(dqn_out):
    dqn_out_clone = torch.clone(dqn_out)
    dqn_out_clone = np.array(dqn_out_clone.detach().cpu())
    act = np.zeros((dqn_out_clone.shape[0], 1, len(dqn_out_clone.shape) - 1))
    for batch in range(len(dqn_out_clone)):
        index = np.unravel_index(dqn_out_clone[batch].argmax(), dqn_out_clone[batch].shape)
        act[batch, 0] = np.array(index)
    act-=1
    return act


def find_hidden_cell_out_of_an_action(act, hidden_state_per_action, cell_state_per_action, out_per_action):
    out = torch.zeros(out_per_action.shape[:2] + out_per_action.shape[-1:])
    hidden_state = torch.zeros(hidden_state_per_action.shape[:2] + hidden_state_per_action.shape[-1:])
    cell_state = torch.zeros(cell_state_per_action.shape[:2] + cell_state_per_action.shape[-1:])
    act = act.long().cpu()
    for batch in range(len(act)):
        out[batch, 0] = out_per_action[batch, 0][tuple(np.array(act[batch, 0]))]
        hidden_state[batch, 0] = hidden_state_per_action[batch, 0][tuple(np.array(act[batch, 0]))]
        cell_state[batch, 0] = cell_state_per_action[batch, 0][tuple(np.array(act[batch, 0]))]
    return hidden_state, cell_state, out


def combine_out(old_out, new_out, atten_size):
    if old_out.shape[1] >= atten_size:
        out = torch.concat((old_out[:, :-1], new_out), 1)
    else:
        out = torch.concat((old_out, new_out), 1)
    return out


def convert_to_array(object):
    if torch.is_tensor(object):
        object = np.array(object.detach().cpu())
    return object


def save_obj(obj, name):
    if not os.path.exists("data"):
        os.mkdir("data")
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_checkpoint(state, filename, dirname='Checkpoint'):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    # filename='Checkpoint_{}.pth.tar'.format(state['episode_count'])
    filepath = os.path.join(dirname, filename)
    torch.save(state, filepath)


def find_latest_checkpoint(dirname='Checkpoint'):
    checkpoints = os.listdir(os.path.join(os.path.dirname(__file__), dirname))
    lastest_checkpoint_count = 0
    if checkpoints:
        for checkpoint in checkpoints:
            current_checkpoint_count = int(checkpoint[11:-8])
            if lastest_checkpoint_count < current_checkpoint_count:
                lastest_checkpoint_count = current_checkpoint_count
                checkpoint_to_load = checkpoint
    return os.path.join(os.path.dirname(__file__), dirname, checkpoint_to_load)


def load_checkpoint(filename, device, dirname='Checkpoint'):
    filepath = os.path.join(dirname, filename)
    checkpoint = torch.load(filepath, map_location=device)

    return checkpoint['model_state_dicts'], checkpoint['optimizer_state_dicts'], checkpoint[
        'episode_count'], checkpoint['epsilon'], checkpoint['epoch_count'], checkpoint['success_count']

def wait_until_present(server, name):
    # print("Waiting for "+name)
    while True:
        if not server.get(name) is None:
            # print(name+" received")
            break
        time.sleep(0.1)

def wait_until_all_received(server, name, num):
    while True:
        if server.llen(name) == num:
            break
        time.sleep(0.1)

# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')

def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, global_grads, mode='grads')

# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])

def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()