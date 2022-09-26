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