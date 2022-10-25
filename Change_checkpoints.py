from utils import load_checkpoint, save_checkpoint
checkpoint=load_checkpoint("Checkpoint_Nav_Human_Play.pth.tar", "cuda")
save_checkpoint({
                    'model_state_dicts': checkpoint['model_state_dicts'],
                    'optimizer_state_dicts': checkpoint['optimizer_state_dicts'],
                    'epsilon': 0.99,
                    "episode_count": 0
                }, filename="Checkpoint_Nav_Human_Play.pth.tar")
