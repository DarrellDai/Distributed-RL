from collections import deque
import random
import numpy as np
from copy import deepcopy


class Memory():

    def __init__(self, memsize, agent_ids):
        self.memsize = memsize
        self.memory = {}
        self.agent_ids = agent_ids
        for id in agent_ids:
            self.memory[id] = deque(maxlen=self.memsize)

    def add_episode(self, episodes):
        for id in episodes:
            self.memory[id].append(episodes[id])
        self.check_dimension()

    def get_batch(self, bsize, time_step, agent_id):
        self.check_dimension()
        batches = []
        memory_long_enough =[]
        for episode in self.memory[agent_id]:
            if len(episode)>=time_step:
                memory_long_enough.append(episode)
        for _ in range(int(np.floor(len(memory_long_enough) / bsize))):
            batches.append([])
        sampled_idx = random.sample(range(len(memory_long_enough)), len(memory_long_enough))
        order = 0
        for batch in batches:
            while len(batch) < bsize and order < len(memory_long_enough):
                point = np.random.randint(0, len(memory_long_enough[sampled_idx[order]]) + 1 - time_step)
                batch.append(memory_long_enough[sampled_idx[order]][point:point + time_step])
                order += 1
        return batches

    def check_dimension(self):
        for id in self.agent_ids:
            for episode in self.memory[id]:
                for t in episode:
                    if len(t[0][0].shape) != 3:
                        raise RuntimeError
