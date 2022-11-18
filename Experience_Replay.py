import _pickle as cPickle
import random
import threading
import time
from collections import deque

import numpy as np
import redis


class Memory():

    def __init__(self, memsize, agent_ids):
        self.memsize = memsize
        self.replay_buffer = {}
        self.agent_ids = agent_ids
        for id in agent_ids:
            self.replay_buffer[id] = deque(maxlen=self.memsize)

    def add_episode(self, episodes):
        clip_flag=False
        for id in self.agent_ids:
            for t in range(len(episodes[id])):
                if episodes[id][t][2]== -1:
                    if episodes[id][t][0][1][0]==episodes[id][t][3][1][0]:
                        raise RuntimeError("The agent is dead, but still showing alive in its observation")
                if episodes[id][t][0][1][0] == 0:
                    clip_flag=True
                    break
            if clip_flag:
                self.replay_buffer[id].append(episodes[id][:t])
            else:
                self.replay_buffer[id].append(episodes[id])
        self.check_dimension()

    # Todo: Change to accept episodes with all length, so some short episode can also be learned
    def get_batch(self, bsize, num_batch, num_learners, time_step):
        self.check_dimension()
        self.check_total_num_vs_needed_for_batch(bsize, num_batch, num_learners)
        batches = []
        for idx_in_batch in range(num_learners):
            batches.append({})
            for id in self.agent_ids:
                batches[idx_in_batch][id] = []
        sampled_idx = random.sample(range(len(self)), bsize * num_batch * num_learners)
        sampled_idx = np.array(sampled_idx).reshape((num_learners, num_batch, bsize))
        for learner_idx in range(num_learners):
            for id in self.agent_ids:
                for batch_idx in range(num_batch):
                    buffer = []
                    for idx_in_batch in range(bsize):
                        try:
                            point = np.random.randint(0, len(self.replay_buffer[id][sampled_idx[
                                learner_idx, batch_idx, idx_in_batch]]) + 1 - time_step)
                            buffer.append(self.replay_buffer[id][sampled_idx[learner_idx, batch_idx, idx_in_batch]][
                                          point:point + time_step])
                        except:
                            buffer.append(self.replay_buffer[id][sampled_idx[learner_idx, batch_idx, idx_in_batch]])
                    batches[learner_idx][id].append(buffer)
        return batches

    def check_dimension(self):
        for id in self.agent_ids:
            for episode in self.replay_buffer[id]:
                for t in episode:
                    if len(t[0][0].shape) != 3:
                        raise RuntimeError("The dimension of the visual observation is wrong")

    def check_total_num_vs_needed_for_batch(self, bsize, num_batch, num_learners):
        if bsize * num_batch * num_learners > len(self):
            raise RuntimeError("Memory is not enough for making the batches")

    def __len__(self):
        return len(self.replay_buffer[self.agent_ids[0]])


class Distributed_Memory(threading.Thread):

    def __init__(self, memsize, agent_ids, connect=redis.Redis(host="localhost")):
        super().__init__()
        self.setDaemon(True)
        self.memsize = memsize
        self._memory = Memory(memsize, agent_ids)
        self._connect = connect
        self._connect.delete("experience")
        self._lock = threading.Lock()

    def run(self):
        while True:
            with self._connect.lock("Update Experience"):
                pipe = self._connect.pipeline()
                pipe.lrange("experience", 0, -1)
                pipe.delete("experience")
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

    def get_batch(self, bsize, num_batch, num_learner, time_step):
        with self._lock:
            return self._memory.get_batch(bsize, num_batch, num_learner, time_step)

    def __len__(self):
        return len(self._memory)
