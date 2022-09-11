import gym
from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment
import matplotlib.pyplot as plt

env_path = 'D:/Unity Projects/Hide and Seek/Env/Hide and Seek'
unity_env = UnityEnvironment(env_path)

env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True)
obs = env.reset()
# {0: agent 0's observation, 1: ...]
for i in range(100):
    actions = {}
    # {0: agent 0's action, 1: ...]
    for id in env.action_space:
        actions[id] = env.action_space[id].sample()
    info = env.step(actions)
# unity_env.close()
