import torch
import numpy as np
import pickle
import torch
import os
import re
import argparse

def train_input_parameters():
    parser=base_input_parameters()
    parser.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                        help="If resume the training or start from scratch")
    parser.add_argument("--checkpoint_to_save", default="Checkpoint.pth.tar", type=str,
                        help="Checkpoint to save")
    parser.add_argument("--batch_size", default=25, type=int, help="Batch size for training")
    parser.add_argument("--time_step", default=20, type=int, help="Length of trajectory to extract for replay buffer")
    parser.add_argument("--learning_rate", "-lr", default=0.00025, type=float, help="Learning rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount Factor")
    parser.add_argument("--initial_epsilon", default=1.0, type=float, help="Initial exploration rate")
    parser.add_argument("--final_epsilon", default=0.1, type=float, help="Final exploration rate")
    parser.add_argument("--epsilon_vanish_rate", default=0.9999, type=float, help="Epsilon Vanish Rate")
    parser.add_argument("--memory_size", default=50, type=int, help="Number of episodes in memory (replay buffer)")
    parser.add_argument("--performance_display_interval", "-pdi", default=20, type=int,
                        help="Number of episodes before displaying the performance of the model")
    parser.add_argument("--checkpoint_save_interval", "-csi", default=25, type=int,
                        help="Number of episodes before saving the checkpoint of the model")
    parser.add_argument("--update_freq", default=5, type=int, help="Number of steps before updating the main model")
    parser.add_argument("--target_update_freq", default=500, type=int,
                        help="Number of steps before updating the target model")
    return parser
def base_input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_to_load", default="Checkpoint.pth.tar", type=str, help="Checkpoint to load")
    parser.add_argument("--device", default=[0], type=int, nargs='+', help="Device Ids for training")
    parser.add_argument("--name", default="CNN_LSTM_DQN", type=str, help="Experiment name for Tensorboard saving path")
    parser.add_argument("--cnn_out_size", default=[500, 500], type=int, nargs='+',
                        help="CNN output size for each agent")
    parser.add_argument("--lstm_hidden_size", default=[512, 512], type=int, nargs='+',
                        help="LSTM hidden size for each agent")
    parser.add_argument("--atten_size", default=[15, 15], type=int, nargs='+', help="Attention size for each agent")
    parser.add_argument("--action_shape", type=int, nargs='+', action='append',
                        help="Action size for each branch. One input per agent")
    parser.add_argument("--act_out_size", default=[32, 32], type=int, nargs='+',
                        help="ActNet output size for each agent")
    parser.add_argument("--env_path", default='../Env/Hide and Seek', type=str, metavar='PATH',
                        help="Path of the Unity environment")
    parser.add_argument("--total_episodes", default=2000, type=int, help="Number of Episodes to play")
    parser.add_argument("--max_steps", default=100, type=int, help="Maximal number of steps for each episode")

    return parser

def import_parameter_from_args(args):
    AGENT_ID = tuple(args.agent_ids)
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
    return AGENT_ID, CNN_OUT_SIZE, LSTM_HIDDEN_SIZE, ACTION_SHAPE, ACTION_OUT_SIZE, ATTEN_SIZE, BATCH_SIZE, TIME_STEP, LR, GAMMA, INITIAL_EPSILON, FINAL_EPSILON, EPSILON_VANISH_RATE, TOTAL_EPSIODES, MAX_STEPS, MEMORY_SIZE, PERFORMANCE_DISPLAY_INTERVAL, CHECKPOINT_SAVE_INTERVAL, UPDATE_FREQ, TARGET_UPDATE_FREQ

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

    return checkpoint['model_state_dicts'], checkpoint['optimizer_state_dicts'], checkpoint['total_steps'], checkpoint[
        'episode_count'], checkpoint['epsilon'], checkpoint['memory']
