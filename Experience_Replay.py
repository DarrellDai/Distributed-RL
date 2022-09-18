from collections import deque
import random
import numpy as np

class Memory():

    def __init__(self, memsize, agent_ids):
        self.memsize = memsize
        self.memory={}
        for id in agent_ids:
            self.memory[id] = deque(maxlen=self.memsize)

    def add_episode(self, episodes):
        for id in episodes:
            self.memory[id].append(episodes[id])

    def get_batch(self, bsize, time_step):
        batch = []
        num_episode_left = bsize
        while num_episode_left>0:
            sampled_epsiodes = random.sample(self.memory, bsize)
            for episode in sampled_epsiodes:
                if len(episode)>=time_step:
                    point = np.random.randint(0, len(episode) + 1 - time_step)
                    batch.append(episode[point:point + time_step])
                else:
                    num_episode_left=bsize-len(batch)
        return batch