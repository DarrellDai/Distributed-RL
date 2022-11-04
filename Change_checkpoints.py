from utils import load_checkpoint, save_checkpoint
import os
import torch
file = "Human_play_Nav_Big_Rocks_No_Waiting_Time_Darrell.pth.tar"
# checkpoint = load_checkpoint(file, "cuda")
# save_checkpoint({
#     'model_state_dicts': checkpoint[0],
#     'optimizer_state_dicts': checkpoint[1],
#     'epsilon': 0.99,
#     'episode_count': 0,
#     'epoch_count': 0,
#     "success_count": 0
# }, filename=file)
filepath = os.path.join('Checkpoint', file)
checkpoint = torch.load(filepath, "cuda")
memory = checkpoint["memory"]
memory.replay_buffer[3]=memory.replay_buffer[1]
del memory.replay_buffer[1]
memory.agent_ids=(3,)
save_checkpoint({
    "memory": memory,
    'id_to_name': {3:"Hider0"}
}, filename=file)