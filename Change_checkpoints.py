from utils import load_checkpoint, save_checkpoint
file="Checkpoint_Nav_Human_Play_RL.pth.tar"
checkpoint=load_checkpoint(file, "cuda")
save_checkpoint({
                    'model_state_dicts': checkpoint[0],
                    'optimizer_state_dicts': checkpoint[1],
                    'epsilon': 0.99,
                    'episode_count': 0,
                    'epoch_count':0
                }, filename=file)
