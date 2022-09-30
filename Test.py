from Pipeline import Pipeline
from utils import base_input_parameters
import yaml

with open("config/Test_Nav.yaml") as file:
    param = yaml.safe_load(file)
args = base_input_parameters().parse_args()
pipeline = Pipeline()
pipeline.initialize_model_and_env(cnn_out_size=param["cnn_out_size"], lstm_hidden_size=param["lstm_hidden_size"],
                                  action_shape=param["action_shape"],
                                  action_out_size=param["action_out_size"], atten_size=param["atten_size"],
                                  device_idx=param["device_idx"], env_path=param["env_path"])

pipeline.test(max_steps=param["max_steps"], total_episodes=param["total_episodes"], checkpoint_to_load=param["checkpoint_to_load"], name_tensorboard=param["name_tensorboard"])
pipeline.env.close()