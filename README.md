# Distributed-RL
Distributed RL for Hide and Seek
## Instruction
### Algorithms
It current supports BC(Behavior Cloning), DQN(Deep Q Network), PPO(Proximal Policy Optimization). Set the hyperparameters in ```.yaml``` files in ```Config/Methods/...```
### Train
#### Training methods
It currently supports **Self-play** and **Imitation Learning**. DQN and PPO is available for **Self-play**, and BC is available for **Imitation Learning**.

The training config can be set in ```.yaml``` files in ```Config/Run/...``` and distributed learning hyperparameters can be set in ```.conf``` files in ```Config/Shell/...```
#### Start training
Run ```. Pipeline/run.sh <Training methods.conf>```
