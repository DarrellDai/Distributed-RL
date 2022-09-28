import keyboard
import numpy as np
import torch
from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment
from utils import base_input_parameters, save_checkpoint, load_checkpoint
from tqdm import tqdm
from copy import deepcopy
from Experience_Replay import Memory
from datetime import datetime
import os

args = base_input_parameters().parse_args()
device = torch.device('cuda:' + str(args.device[0]) if torch.cuda.is_available() else 'cpu')
checkpoint = "Human_play_" + datetime.now().isoformat()[:10] + "_" + datetime.now().isoformat()[
                                                                     11:13] + "-" + datetime.now().isoformat()[
                                                                                    14:16] + ".pth.tar"

unity_env = UnityEnvironment(args.env_path)
env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
agent_ids = (0, 1)
max_step = 50
mem = Memory(memsize=args.total_episodes, agent_ids=agent_ids)
exit = False
for _ in tqdm(range(args.total_episodes)):
    prev_obs = env.reset()
    step_count = 0
    local_memory = {0: [], 1: []}
    done = False
    while step_count<max_step and not done:
        act = {}
        # {0: agent 0's action, 1: ...]
        for id in agent_ids:
            act[id] = np.zeros(2)
        if keyboard.is_pressed("down"):
            act[1][0] = -1
        elif keyboard.is_pressed("up"):
            act[1][0] = 1
        else:
            act[1][0] = 0
        if keyboard.is_pressed("right"):
            act[1][1] = 1
        elif keyboard.is_pressed("left"):
            act[1][1] = -1
        else:
            act[1][1] = 0

        if keyboard.is_pressed("s"):
            act[0][0] = -1
        elif keyboard.is_pressed("w"):
            act[0][0] = 1
        else:
            act[0][0] = 0
        if keyboard.is_pressed("d"):
            act[0][1] = 1
        elif keyboard.is_pressed("a"):
            act[0][1] = -1
        else:
            act[0][1] = 0
        if keyboard.is_pressed("esc"):
            exit = True
            break
        obs_dict, reward_dict, done_dict, info_dict = env.step(act)
        step_count += 1
        done = done_dict["__all__"]
        for id in agent_ids:
            local_memory[id].append(
                (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]), deepcopy(obs_dict[id])))
        prev_obs = obs_dict
    if exit:
        save_checkpoint({"memory": mem}, filename=checkpoint)
        env.close()
        break

    mem.add_episode(local_memory)

# filepath = os.path.join('Checkpoint', 'Human_play_2022-09-28_15-28.pth.tar')
# checkpoint = torch.load(filepath, device)
