from Pipeline import Pipeline
from utils import input_parameters

args = input_parameters()
pipeline = Pipeline(args.cnn_out_size, args.lstm_hidden_size, args.action_shape, args.act_out_size, args.atten_size,
                    args.device)
pipeline.initialize_model_and_env(args.env_path, args.learning_rate)
mem, criterion, optimizer, target_model, writer = pipeline.initialize_training(args.memory_size, args.learning_rate,
                                                                               args.name)
if args.resume:
    target_model, optimizer, total_steps, start_episode, episode_count, epsilon, mem = pipeline.resume_training(
        args.checkpoint_to_load, optimizer, target_model)
else:
    epsilon = args.initial_epsilon
    total_steps = 0
    episode_count = 0
    start_episode = episode_count
    pipeline.fill_memory_with_random_walk(mem, max_step=args.max_steps)
pipeline.train(start_episode, episode_count, args.total_episodes, total_steps, args.gamma, epsilon, args.final_epsilon,
               args.epsilon_vanish_rate, args.max_steps, target_model,
               mem, args.batch_size, args.time_step, args.learning_rate, args.target_update_freq, args.update_freq,
               optimizer, criterion,
               args.performance_display_interval, args.checkpoint_save_interval,
               writer, args.checkpoint_to_save)
