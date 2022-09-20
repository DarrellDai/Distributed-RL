from collections import deque
import random
import numpy as np

class Memory():

    def __init__(self, memsize, agent_ids):
        self.memsize = memsize
        self.memory={}
        self.agent_ids=agent_ids
        for id in agent_ids:
            self.memory[id] = deque(maxlen=self.memsize)

    def add_episode(self, episodes):
        for id in episodes:
            self.memory[id].append(episodes[id])
        self.check_dimension()

    def get_batch(self, bsize, time_step, agent_id):
        self.check_dimension()
        batch = []
        num_episode_left = bsize
        while num_episode_left>0:
            sampled_epsiodes = random.sample(self.memory[agent_id], num_episode_left)
            for episode in sampled_epsiodes:
                if len(episode)>=time_step:
                    point = np.random.randint(0, len(episode) + 1 - time_step)
                    batch.append(episode[point:point + time_step])
                    num_episode_left-=1
        return batch
    def check_dimension(self):
        for id in self.agent_ids:
            for episode in self.memory[id]:
                for t in episode:
                    if len(t[0][0].shape)!=3:
                        raise RuntimeError