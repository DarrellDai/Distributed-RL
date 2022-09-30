from Pipeline import Pipeline
from utils import train_input_parameters

args = train_input_parameters().parse_args()
pipeline = Pipeline()
pipeline.initialize_model_and_env(args.cnn_out_size, args.lstm_hidden_size, args.action_shape, args.act_out_size, args.atten_size,
                    args.device, args.env_path)
pipeline.env.close()
mem, criterion, optimizer, target_model = pipeline.initialize_training(args.memory_size, args.learning_rate)

pipeline.train_from_human_play()
