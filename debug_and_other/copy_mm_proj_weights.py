import torch

# Path to the original model weights
source_bin_path = '/home/ai_center/ai_users/roeibenzion/VLM-FGA/LLaVA_converter/checkpoints/llava-v1.5-7b-pretrain/checkpoint-4500/mm_projector.bin'  # or 'checkpoint-*/pytorch_model.bin'
target_bin_path = 'mm_projector_only.bin'

# Load the full state dict
state_dict = torch.load(source_bin_path, map_location='cpu')

# Filter keys that contain 'mm_projector'
filtered_state_dict = {
    k: v for k, v in state_dict.items() if 'mm_projector' in k
}

# Optional: print keys to verify
print("Filtered keys:")
for k in filtered_state_dict:
    print(k)

# Save to new .bin file
torch.save(filtered_state_dict, target_bin_path)
print(f"Saved {len(filtered_state_dict)} keys to {target_bin_path}")
