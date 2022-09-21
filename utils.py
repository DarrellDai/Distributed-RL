import torch
import numpy as np
import pickle
import torch
import os

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
    if not os.path.exists("data"):
        os.mkdir("data")
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_checkpoint(state, dirname='Checkpoint'):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filename = 'Checkpoint.pth.tar'
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
                checkpoint_to_load=checkpoint
    return os.path.join(os.path.dirname(__file__), dirname, checkpoint_to_load)
def load_checkpoint(file):
    checkpoint = torch.load(file)

    return checkpoint['model_state_dicts'], checkpoint['optimizer_state_dicts'] ,checkpoint['total_steps'], checkpoint['episode_count'], checkpoint['epsilon'], checkpoint['memory'], checkpoint['loss_stat'], checkpoint['reward_stat']