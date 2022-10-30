from collections import deque
import random
import numpy as np
from copy import deepcopy
import threading
import redis
import time
import _pickle as cPickle
class Memory():

    def __init__(self, memsize, agent_ids):
        self.memsize = memsize
        self.replay_buffer = {}
        self.agent_ids = agent_ids
        for id in agent_ids:
            self.replay_buffer[id] = deque(maxlen=self.memsize)

    def add_episode(self, episodes):
        for id in self.agent_ids:
            self.replay_buffer[id].append(episodes[id])
        self.check_dimension()

    def get_batch(self, bsize, time_step, agent_id):
        self.check_dimension()
        batches = []
        memory_long_enough =[]
        for episode in self.replay_buffer[agent_id]:
            if len(episode)>=time_step:
                memory_long_enough.append(episode)
        for _ in range(int(np.ceil(len(memory_long_enough) / bsize))):
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
            for episode in self.replay_buffer[id]:
                for t in episode:
                    if len(t[0][0].shape) != 3:
                        raise RuntimeError

    def __len__(self):
        return len(self.replay_buffer[self.agent_ids[0]])

class Distributed_Memory(threading.Thread):

    def __init__(self, memsize, agent_ids, connect=redis.Redis(host="localhost")):
        super().__init__()
        self.setDaemon(True)
        self.memsize=memsize
        self._memory = Memory(memsize, agent_ids)
        self._connect = connect
        self._connect.delete("experience")
        self._lock = threading.Lock()

    def run(self):
        while True:
            pipe = self._connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            memories = pipe.execute()[0]
            if not memories is None:
                for memory in memories:
                    load_memory = cPickle.loads(memory)
                    with self._lock:
                        for i in range(len(load_memory)):
                            episode = {}
                            for id in load_memory.agent_ids:
                                episode[id] = load_memory.replay_buffer[id][i]
                            self._memory.add_episode(episode)
            time.sleep(0.01)

    def get_batch(self, bsize, time_step, agent_id):
        with self._lock:
            return self._memory.get_batch(bsize, time_step, agent_id)

    def __len__(self):
        return len(self._memory)
