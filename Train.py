import numpy as np
import random
import torch.nn as nn
import torch
from Network import Network
from Experience_Replay import Memory
from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment
from copy import deepcopy
from utils import find_optimal_action, save_obj, convert_to_array

AGENT_ID = (0, 1)
CNN_OUT_SIZE = {0: 512, 1: 512}
LSTM_HIDDEN_SIZE = {0: 512, 1: 512}
ATTEN_SIZE = {0: 2, 1: 2}
ACTION_SHAPE = {0: (3, 3), 1: (3, 3)}
BATCH_SIZE = 32
TIME_STEP = 15
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
TOTAL_EPSIODES = 20000
MAX_STEPS = 200
MEMORY_SIZE = 3000
UPDATE_FREQ = 5
PERFORMANCE_SAVE_INTERVAL = 500
TARGET_UPDATE_FREQ = 10000  # steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mem = Memory(memsize=MEMORY_SIZE, agent_ids=AGENT_ID)
main_model = {}
target_model = {}
optimizer = {}
for id in AGENT_ID:
    main_model[id] = Network(cnn_out_size=CNN_OUT_SIZE[id], lstm_hidden_size=LSTM_HIDDEN_SIZE[id],
                             atten_size=ATTEN_SIZE[id], action_shape=ACTION_SHAPE[id]).float().to(device)
    target_model[id] = Network(cnn_out_size=CNN_OUT_SIZE[id], lstm_hidden_size=LSTM_HIDDEN_SIZE[id],
                               atten_size=ATTEN_SIZE[id], action_shape=ACTION_SHAPE[id]).float().to(device)

    target_model[id].load_state_dict(main_model[id].state_dict())
    optimizer[id] = torch.optim.Adam(main_model[id].parameters(), lr=0.00025)

criterion = nn.MSELoss()
env_path = 'D:/Unity Projects/Hide and Seek/Env/Hide and Seek'
unity_env = UnityEnvironment(env_path)

env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)

# Fill memory
for i in range(MEMORY_SIZE):

    prev_obs = env.reset()
    step_count = 0
    local_memory = {}
    for id in AGENT_ID:
        local_memory[id] = []
        prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(device)
    done = False
    while step_count < MAX_STEPS and not done:
        act = {}
        # {0: agent 0's action, 1: ...]
        for id in AGENT_ID:
            act[id] = env.action_space[id].sample()
        obs_dict, reward_dict, done_dict, info_dict = env.step(act)
        step_count += 1
        done = done_dict["__all__"]
        for id in AGENT_ID:
            local_memory[id].append(
                (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]), deepcopy(obs_dict[id])))

        prev_obs = obs_dict

    mem.add_episode(local_memory)

print('Populated with %d Episodes' % (len(mem.memory[0])))

# Start Algorithm
epsilon = INITIAL_EPSILON
loss_stat = {}
reward_stat = {}
total_steps = 0
for id in AGENT_ID:
    loss_stat[id] = []
    reward_stat[id] = []
episode_count = 0
for episode in range(TOTAL_EPSIODES):
    episode_count += 1
    total_reward = {}
    step_count = 0
    local_memory = {}
    hidden_state = {}
    cell_state = {}
    lstm_out = {}
    prev_obs = env.reset()
    for id in AGENT_ID:
        total_reward[id] = 0
        local_memory[id] = []

        hidden_state[id], cell_state[id], lstm_out[id] = main_model[id].lstm.init_hidden_states_and_outputs(bsize=1)
    done = False
    while step_count < MAX_STEPS and not done:

        step_count += 1
        total_steps += 1
        act = {}
        if np.random.rand(1) < epsilon:
            for id in AGENT_ID:
                act[id] = ()
                for n in main_model[id].action_shape:
                    act[id] += (np.random.randint(0, n),)
                act[id] = torch.tensor(act[id]).reshape(1, 1, len(act[id])).to(device)
                prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(device)
                prev_obs[id][0] = prev_obs[id][0].reshape(1, 1, prev_obs[id][0].shape[0], prev_obs[id][0].shape[1],
                                                          prev_obs[id][0].shape[2])
                model_out = main_model[id].forward(prev_obs[id][0], act[id], bsize=1, hidden_state=hidden_state[id],
                                                   cell_state=cell_state[id], lstm_out=lstm_out[id])

                hidden_state[id] = model_out[1][0]
                cell_state[id] = model_out[1][1]
                lstm_out[id] = model_out[0]


        else:
            for id in AGENT_ID:
                prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(device)
                prev_obs[id][0] = prev_obs[id][0].reshape(1, 1, prev_obs[id][0].shape[0], prev_obs[id][0].shape[1],
                                                          prev_obs[id][0].shape[2])
                model_out = main_model[id].forward(prev_obs[id][0], act=torch.zeros((1, 0, len(ACTION_SHAPE))), bsize=1,
                                                   hidden_state=hidden_state[id],
                                                   cell_state=cell_state[id], lstm_out=lstm_out[id])

                act[id] = torch.from_numpy(find_optimal_action(model_out[2]))
                model_out = main_model[id].forward(prev_obs[id][0], act[id], bsize=1, hidden_state=hidden_state[id],
                                                   cell_state=cell_state[id], lstm_out=lstm_out[id])
                hidden_state[id] = model_out[1][0]
                cell_state[id] = model_out[1][1]
                lstm_out[id] = model_out[0]

        obs_dict, reward_dict, done_dict, info_dict = env.step(act)
        done = done_dict["__all__"]

        for id in AGENT_ID:
            total_reward[id] += reward_dict[id]

            prev_obs[id][0] = prev_obs[id][0].reshape(prev_obs[id][0].shape[2], prev_obs[id][0].shape[3],
                                                      prev_obs[id][0].shape[4])
            act[id] = act[id].reshape(-1)

            local_memory[id].append(
                (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]), deepcopy(obs_dict[id])))

        prev_obs = deepcopy(obs_dict)

        if (total_steps % TARGET_UPDATE_FREQ) == 0:
            for id in AGENT_ID:
                target_model[id].load_state_dict(main_model[id].state_dict())

        if (total_steps % UPDATE_FREQ) == 0:

            for id in AGENT_ID:
                print("start training" + str(id))
                hidden_batch, cell_batch, out_batch = main_model[id].lstm.init_hidden_states_and_outputs(
                    bsize=BATCH_SIZE)
                batch = mem.get_batch(bsize=BATCH_SIZE, time_step=TIME_STEP, agent_id=id)
                current_visual_obs = []
                current_vector_obs = []
                act = []
                rewards = []
                next_visual_obs = []
                next_vector_obs = []

                for b in batch:
                    cvis, cves, ac, rw, nvis, nves = [], [], [], [], [], []
                    for element in b:
                        cvis.append(convert_to_array(element[0][0]))
                        cves.append(convert_to_array(element[0][1][list(element[0][1].keys())[0]]))
                        ac.append(convert_to_array(element[1]))
                        rw.append(convert_to_array(element[2]))
                        nvis.append(convert_to_array(element[3][0]))
                        nves.append(convert_to_array(element[3][1][list(element[0][1].keys())[0]]))
                    current_visual_obs.append(cvis)
                    current_vector_obs.append(cves)
                    act.append(ac)
                    rewards.append(rw)
                    next_visual_obs.append(nvis)
                    next_vector_obs.append(nves)

                current_visual_obs = np.array(current_visual_obs)
                current_vector_obs = np.array(current_vector_obs)
                act = np.array(act)
                rewards = np.array(rewards)
                next_visual_obs = np.array(next_visual_obs)
                next_vector_obs = np.array(next_vector_obs)

                current_visual_obs = torch.from_numpy(current_visual_obs).float().to(device)
                act = torch.from_numpy(act).long().to(device)
                rewards = torch.from_numpy(rewards).float().to(device)
                next_visual_obs = torch.from_numpy(next_visual_obs).float().to(device)
                visual_obs = torch.concat((current_visual_obs, next_visual_obs[:, -1:]), 1)
                Q_next_max = torch.zeros(BATCH_SIZE).float().to(device)
                for batch_idx in range(BATCH_SIZE):
                    if next_vector_obs[batch_idx][0][0] == 1:
                        _, _, Q_next, _, _ = target_model[id].forward(visual_obs[batch_idx:batch_idx + 1],
                                                                      act[batch_idx:batch_idx + 1],
                                                                      bsize=1,
                                                                      hidden_state=hidden_batch, cell_state=cell_batch,
                                                                      lstm_out=out_batch)
                        Q_next_max[batch_idx] = torch.max(Q_next.reshape(-1))
                target_values = rewards[:, TIME_STEP - 1] + (GAMMA * Q_next_max)
                target_values = target_values.float()
                _, _, Q_s, _, _ = main_model[id].forward(current_visual_obs, act, bsize=BATCH_SIZE,
                                                         hidden_state=hidden_batch, cell_state=cell_batch,
                                                         lstm_out=out_batch)

                Q_s_a = Q_s[0, 0]
                loss = criterion(Q_s_a, target_values)

                #  save performance measure
                loss_stat[id].append(loss.item())

                # make previous grad zero
                optimizer[id].zero_grad()

                # backward
                loss.backward()

                # update params
                optimizer[id].step()
    # save performance measure
    mem.add_episode(local_memory)

    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / TOTAL_EPSIODES

    if (episode + 1) % PERFORMANCE_SAVE_INTERVAL == 0:
        perf = {}
        perf['loss'] = loss_stat
        perf['total_reward'] = reward_stat
        save_obj(name='LSTM_POMDP_V4', obj=perf)

    # print('Episode : ',episode+1,'Epsilon : ',epsilon,'Reward : ',total_reward,)
