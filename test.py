from Pipeline import Pipeline
from utils import base_input_parameters

args = base_input_parameters().parse_args()
pipeline = Pipeline(args.cnn_out_size, args.lstm_hidden_size, args.action_shape, args.act_out_size, args.atten_size,
                    args.device)
pipeline.initialize_model_and_env(args.env_path)

pipeline.test(args.max_steps, args.total_episodes, args.checkpoint_to_load, args.name)
