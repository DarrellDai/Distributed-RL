from Pipeline import Pipeline
import yaml

with open("config/Test_Nav.yaml") as file:
    param = yaml.safe_load(file)
pipeline = Pipeline()
pipeline.initialize_env(env_path=param["env_path"])
pipeline.initialize_model(cnn_out_size=param["cnn_out_size"], lstm_hidden_size=param["lstm_hidden_size"],
                          action_shape=param["action_shape"],
                          action_out_size=param["action_out_size"], atten_size=param["atten_size"],
                          device_idx=param["device_idx"])

pipeline.test(max_steps=param["max_steps"], total_episodes=param["total_episodes"], checkpoint_to_load=param["checkpoint_to_load"], name_tensorboard=param["name_tensorboard"])
pipeline.env.close()