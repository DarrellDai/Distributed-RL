import gym
from unity_wrappers.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

env_path = 'D:/Unity Projects/Hide and Seek/Env/Hide and Seek'
unity_env = UnityEnvironment(env_path)

env = UnityToGymWrapper(unity_env=unity_env, uint8_visual=True)
unity_env.close()
