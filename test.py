import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from Network import Network
from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment
from copy import deepcopy

import os

from utils import input_parameters, import_parameter_from_args, find_name_of_agents, find_optimal_action, \
    find_hidden_cell_out_of_an_action, combine_out, load_checkpoint
from tqdm import tqdm

args = input_parameters()
AGENT_ID, CNN_OUT_SIZE, LSTM_HIDDEN_SIZE, ACTION_SHAPE, \
ACTION_OUT_SIZE, ATTEN_SIZE, BATCH_SIZE, TIME_STEP, LR, \
GAMMA, INITIAL_EPSILON, FINAL_EPSILON, EPSILON_VANISH_RATE, \
TOTAL_EPSIODES, MAX_STEPS, MEMORY_SIZE, PERFORMANCE_DISPLAY_INTERVAL, \
CHECKPOINT_SAVE_INTERVAL, UPDATE_FREQ, TARGET_UPDATE_FREQ, MAX_LOSS_STAT_LEN, \
MAX_REWARD_STAT_LEN = import_parameter_from_args(args)
INITIAL_EPSILON, INITIAL_EPSILON = 0, 0
device = torch.device('cuda:' + str(args.device[0]) if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(os.path.join("runs", "test", args.name))

unity_env = UnityEnvironment(args.env_path)
env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
id_to_name = find_name_of_agents(env.agent_id_to_behaviour_name, AGENT_ID)
main_model = {}

for id in AGENT_ID:
    main_model[id] = Network(cnn_out_size=CNN_OUT_SIZE[id], lstm_hidden_size=LSTM_HIDDEN_SIZE[id],
                             atten_size=ATTEN_SIZE[id], action_space_shape=ACTION_SHAPE[id],
                             action_out_size=ACTION_OUT_SIZE[id])
    main_model[id] = nn.DataParallel(main_model[id], device_ids=args.device)

checkpoint_to_load = os.path.join('Checkpoint', 'Checkpoint.pth.tar')
model_state_dicts, optimizer_state_dicts, _, _, _, _, _, _ = load_checkpoint(
    checkpoint_to_load, 'cuda:' + str(args.device[0]))
for id in AGENT_ID:
    main_model[id].load_state_dict(model_state_dicts[id])

reward_stat = {}
total_steps = 0
episode_count = 0
for id in AGENT_ID:
    reward_stat[id] = []
# Start Algorithm
for episode in tqdm(range(TOTAL_EPSIODES)):
    episode_count += 1
    total_reward = {}
    step_count = 0
    hidden_state = {}
    cell_state = {}
    lstm_out = {}
    prev_obs = env.reset()
    alive = {}

    for id in AGENT_ID:
        total_reward[id] = 0
        alive[id] = True
        hidden_state[id], cell_state[id], lstm_out[id] = main_model[id].module.lstm.init_hidden_states_and_outputs(
            bsize=1)
    done = False
    while step_count < MAX_STEPS and not done:

        step_count += 1
        total_steps += 1
        act = {}
        with torch.no_grad():
            for id in AGENT_ID:
                prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(device)
                prev_obs[id][0] = prev_obs[id][0].reshape(1, 1, prev_obs[id][0].shape[0], prev_obs[id][0].shape[1],
                                                          prev_obs[id][0].shape[2])
                lo, (hs, cs), dqn_out, out_per_action, (
                    hidden_state_per_action, cell_state_per_action) = main_model[id](prev_obs[id][0],
                                                                                     act=torch.zeros((1,
                                                                                                      0,
                                                                                                      len(ACTION_SHAPE))).to(
                                                                                         device),
                                                                                     hidden_state=
                                                                                     hidden_state[id],
                                                                                     cell_state=
                                                                                     cell_state[id],
                                                                                     lstm_out=lstm_out[
                                                                                         id])

                act[id] = torch.from_numpy(find_optimal_action(dqn_out))
                hidden_state[id], cell_state[id], lo_new = find_hidden_cell_out_of_an_action(act[id],
                                                                                             hidden_state_per_action,
                                                                                             cell_state_per_action,
                                                                                             out_per_action)
                lstm_out[id] = combine_out(lo, lo_new.to(device), ATTEN_SIZE[id])
                hidden_state[id] = hidden_state[id].to(device)
                cell_state[id] = cell_state[id].to(device)
            obs_dict, reward_dict, done_dict, info_dict = env.step(act)

            done = done_dict["__all__"]

            for id in AGENT_ID:
                total_reward[id] += reward_dict[id]

                prev_obs[id][0] = prev_obs[id][0].reshape(prev_obs[id][0].shape[2], prev_obs[id][0].shape[3],
                                                          prev_obs[id][0].shape[4])
                act[id] = act[id].reshape(-1)

            prev_obs = deepcopy(obs_dict)

    for id in AGENT_ID:
        writer.add_scalar(id_to_name[id] + ": Reward/train", total_reward[id], episode_count)
    writer.flush()

    if (episode + 1) % PERFORMANCE_DISPLAY_INTERVAL == 0:
        print('\n Episode: [%d | %d] \n' % (episode, TOTAL_EPSIODES))
        for id in AGENT_ID:
            print('\n Agent %d, Reward: %f \n' % (id, total_reward[id]))
