import os

# Run local debug
os.system(
    'python Train.py --env_path "D:/Unity Projects/Hide and Seek/Env/Hide and Seek" --action_shape 3 3 --action_shape 3 3 \
    --resume --atten_size 2 2 --batch_size 6 --time_step 15 --total_episodes 30 --max_steps 30 \
    --memory_size 10 --update_freq 1 -pdi 2 -csi 2 --target_update_freq 90')

# Run server training
os.system(
    'python Train.py --device 3 4 5 6 7 --action_shape 3 3 --action_shape 3 3 --resume')

# Run local training
os.system(
    'python Train.py --env_path "D:/Unity Projects/Hide and Seek/Env/Hide and Seek" \
    --action_shape 3 3 --action_shape 3 3 --resume')

# Run server training
os.system('python Test.py --env_path "D:/Unity Projects/Hide and Seek/Env/Hide and Seek" --action_shape 3 3 --action_shape 3 3 --total_episode 30')

# Run local debug (single agent)
os.system('python Train.py --env_path "D:/Unity Projects/Hide and Seek/Env_Nav/Hide and Seek" --action_shape 3 3 --atten_size 2 \
          --batch_size 6 --time_step 15 --total_episodes 30 --max_steps 30 --memory_size 10 \
          --update_freq 1 -pdi 2 -csi 2 --target_update_freq 90 \
          --checkpoint_to_load "Checkpoint_Nav_1.pth.tar" --no-resume \
          --checkpoint_to_save "Checkpoint_Nav_1.pth.tar" \
          --cnn_out_size 500 --lstm_hidden_size 512 --act_out_size 32')

# Run server training (single agent)
os.system('python Train.py --env_path "../Env_Nav/Hide and Seek" --action_shape 3 3 \
          --checkpoint_to_load "Checkpoint_Nav.pth.tar" --no-resume \
          --checkpoint_to_save "Checkpoint_Nav.pth.tar" \
          --cnn_out_size 500 --lstm_hidden_size 512 \
          --atten_size 15 --act_out_size 32 --device 3 4 5 6 7 \
          --name CNN_LSTM_DQN_Nav')