import torch
import numpy as np
import pickle
import torch
import os
import re


def import_parameter_from_args(args):
    AGENT_ID = tuple(args.agent_id)
    CNN_OUT_SIZE = {}
    LSTM_HIDDEN_SIZE = {}
    ACTION_SHAPE = {}
    ACTION_OUT_SIZE = {}
    ATTEN_SIZE = {}
    for idx in range(len(AGENT_ID)):
        CNN_OUT_SIZE[AGENT_ID[idx]] = args.cnn_out_size[idx]
        LSTM_HIDDEN_SIZE[AGENT_ID[idx]] = args.lstm_hidden_size[idx]
        ACTION_SHAPE[AGENT_ID[idx]] = tuple(args.action_shape[idx])
        ATTEN_SIZE[AGENT_ID[idx]] = args.atten_size[idx]
        ACTION_OUT_SIZE[AGENT_ID[idx]] = args.act_out_size[idx]

    BATCH_SIZE = args.batch_size
    TIME_STEP = args.time_step
    LR = args.learning_rate
    GAMMA = args.gamma
    INITIAL_EPSILON = args.initial_epsilon
    FINAL_EPSILON = args.final_epsilon
    EPSILON_VANISH_RATE = args.epsilon_vanish_rate
    TOTAL_EPSIODES = args.total_episodes
    MAX_STEPS = args.max_steps
    MEMORY_SIZE = args.memory_size
    PERFORMANCE_DISPLAY_INTERVAL = args.performance_display_interval
    CHECKPOINT_SAVE_INTERVAL = args.checkpoint_save_interval
    UPDATE_FREQ = args.update_freq
    TARGET_UPDATE_FREQ = args.target_update_freq
    MAX_LOSS_STAT_LEN = args.max_loss_stat_len
    MAX_REWARD_STAT_LEN = args.max_reward_stat_len
    return AGENT_ID, CNN_OUT_SIZE, LSTM_HIDDEN_SIZE, ACTION_SHAPE, ACTION_OUT_SIZE, ATTEN_SIZE, BATCH_SIZE, TIME_STEP, LR, GAMMA, INITIAL_EPSILON, FINAL_EPSILON, EPSILON_VANISH_RATE, TOTAL_EPSIODES, MAX_STEPS, MEMORY_SIZE, PERFORMANCE_DISPLAY_INTERVAL, CHECKPOINT_SAVE_INTERVAL, UPDATE_FREQ, TARGET_UPDATE_FREQ, MAX_LOSS_STAT_LEN, MAX_REWARD_STAT_LEN


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
    return act


def find_hidden_cell_out_of_an_action(act, hidden_state_per_action, cell_state_per_action, out_per_action):
    out = torch.zeros(out_per_action.shape[:2] + out_per_action.shape[-1:])
    hidden_state = torch.zeros(hidden_state_per_action.shape[:2] + hidden_state_per_action.shape[-1:])
    cell_state = torch.zeros(cell_state_per_action.shape[:2] + cell_state_per_action.shape[-1:])
    act_clone = torch.clone(act)
    act_clone = act_clone.long()
    for batch in range(len(act)):
        out[batch, 0] = out_per_action[batch, 0][tuple(np.array(act_clone[batch, 0]))]
        hidden_state[0, batch] = hidden_state_per_action[batch, 0][tuple(np.array(act_clone[batch, 0]))]
        cell_state[0, batch] = cell_state_per_action[batch, 0][tuple(np.array(act_clone[batch, 0]))]
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
                checkpoint_to_load = checkpoint
    return os.path.join(os.path.dirname(__file__), dirname, checkpoint_to_load)


def load_checkpoint(file):
    checkpoint = torch.load(file)

    return checkpoint['model_state_dicts'], checkpoint['optimizer_state_dicts'], checkpoint['total_steps'], checkpoint[
        'episode_count'], checkpoint['epsilon'], checkpoint['memory'], checkpoint['loss_stat'], checkpoint[
               'reward_stat']
