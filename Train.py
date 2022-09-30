from Pipeline import Pipeline
from Experience_Replay import Memory
import yaml

with open("config/Train_Nav_Local.yaml") as file:
    param = yaml.safe_load(file)

pipeline = Pipeline()
pipeline.initialize_env(env_path=param["env_path"])
pipeline.initialize_model(cnn_out_size=param["cnn_out_size"], lstm_hidden_size=param["lstm_hidden_size"],
                          action_shape=param["action_shape"],
                          action_out_size=param["action_out_size"], atten_size=param["atten_size"],
                          device_idx=param["device_idx"])
criterion, optimizer, target_model = pipeline.initialize_training(learning_rate=param["learning_rate"])
memory = Memory(memsize=param["memory_size"], agent_ids=pipeline.agent_ids)

if param["resume"]:
    target_model, optimizer, total_steps, start_episode, episode_count, epsilon, memory = pipeline.resume_training(
        checkpoint_to_load=param["checkpoint_to_load"], optimizer=optimizer, target_model=target_model)
else:
    epsilon = param["initial_epsilon"]
    total_steps = 0
    episode_count = 0
    start_episode = episode_count
    pipeline.fill_memory_with_random_walk(memory, max_steps=param["max_steps"])

pipeline.train(start_episode=start_episode, episode_count=episode_count, total_episodes=param["total_episodes"],
               total_steps=total_steps,
               gamma=param["gamma"], epsilon=epsilon, final_epsilon=param["final_epsilon"],
               epsilon_vanish_rate=param["epsilon_vanish_rate"], max_steps=param["max_steps"],
               target_model=target_model,
               memory=memory, batch_size=param["batch_size"], time_step=param["time_step"],
               learning_rate=param["learning_rate"],
               target_update_freq=param["target_update_freq(steps)"], update_freq=param["update_freq"], optimizer=optimizer,
               criterion=criterion,
               performance_display_interval=param["performance_display_interval"],
               checkpoint_save_interval=param["checkpoint_save_interval"],
               checkpoint_to_save=param["checkpoint_to_save"], name_tensorboard=param["name_tensorboard"])
pipeline.env.close()
