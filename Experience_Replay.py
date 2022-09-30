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
        for _ in range(int(np.ceil(self.memsize / bsize))):
            batches.append([])
        sampled_idx = random.sample(range(len(self.memory[agent_id])), len(self.memory[agent_id]))
        order = 0
        for batch in batches:
            while len(batch) < bsize and order < len(self.memory[agent_id]):
                if len(self.memory[agent_id][sampled_idx[order]]) >= time_step:
                    point = np.random.randint(0, len(self.memory[agent_id][sampled_idx[order]]) + 1 - time_step)
                    batch.append(self.memory[agent_id][sampled_idx[order]][point:point + time_step])
                else:
                    batch.append(self.memory[agent_id][sampled_idx[order]])
                order += 1
        return batches

    def check_dimension(self):
        for id in self.agent_ids:
            for episode in self.memory[id]:
                for t in episode:
                    if len(t[0][0].shape) != 3:
                        raise RuntimeError
