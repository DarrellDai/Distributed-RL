from Pipeline import Pipeline
import os
import yaml
import torch
import numpy as np
from utils import load_checkpoint, save_checkpoint

with open("Config/Train_Nav_Human_Play_Local.yaml") as file:
    param = yaml.safe_load(file)

pipeline = Pipeline()
pipeline.agent_ids=(1,)
pipeline.id_to_name={1:"Hider 0"}
pipeline.initialize_model(cnn_out_size=param["cnn_out_size"], lstm_hidden_size=param["lstm_hidden_size"],
                          action_shape=param["action_shape"],
                          action_out_size=param["action_out_size"], atten_size=param["atten_size"],
                          device_idx=param["device_idx"])
criterion, optimizer, target_model = pipeline.initialize_training(learning_rate=param["learning_rate"])
# _, _, _, _, _, memory = load_checkpoint(
#             "Checkpoint_Nav_1.pth.tar", pipeline.device)
filepath = os.path.join('Checkpoint', param["memory_to_load"])
checkpoint = torch.load(filepath, pipeline.device)
memory=checkpoint["memory"]
# for id in memory.agent_ids:
#     for episode in memory.memory[id]:
#         for point in episode:
#             point[0][0] = np.array(point[0][0].cpu())
#             point[3][0] = np.array(point[3][0].cpu())
# save_checkpoint({"memory": memory}, filename=param["memory_to_load"])
if param["resume"]:
    target_model, optimizer, _, _, _, _, _ = pipeline.resume_training(
        checkpoint_to_load=param["checkpoint_to_load"], optimizer=optimizer, target_model=target_model)

pipeline.train_from_human_play(batch_size=param["batch_size"], time_step=param["time_step"], gamma=param["gamma"], memory=memory, criterion=criterion,
                           optimizer=optimizer, learning_rate=param["learning_rate"], target_model=target_model,
                           name_tensorboard=param["name_tensorboard"], total_epochs=param["total_epochs"],
                           target_update_freq=param["target_update_freq(epochs)"], checkpoint_save_interval=param["checkpoint_save_interval"],
                           checkpoint_to_save=param["checkpoint_to_save"])
pipeline.env.close()
