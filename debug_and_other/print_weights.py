import torch

# Replace this with your actual path to the .bin model file
model_path = '/home/ai_center/ai_users/roeibenzion/VLM-FGA/LLaVA_converter/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin'

# Load the state dictionary from the .bin file
state_dict = torch.load(model_path, map_location='cpu')

# Print all weight keys
print("Weight keys in the model:")
for key in state_dict.keys():
    print(key)
