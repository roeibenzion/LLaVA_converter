import torch
import os

# Paths
pretrain_mm_mlp_adapter = "/home/ai_center/ai_users/roeibenzion/VLM-FGA/LLaVA_converter/checkpoints/llava-v1.5-7b-pretrain/checkpoint-4500/mm_projector.bin"
path_to_weights = pretrain_mm_mlp_adapter  # same file

# Dummy components (must match your real model structure for actual tests)
class DummyMMProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 8)

class DummyFGA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atten1 = torch.nn.Linear(8, 4)
        self.atten2 = torch.nn.Linear(4, 2)

# Dummy model
class DummyModel:
    def __init__(self):
        self.mm_projector = DummyMMProjector()
        self.vision_tower = type("VisionTower", (), {"config": type("Cfg", (), {"hidden_size": 64})})()
        self.config = type("Cfg", (), {"hidden_size": 128})()

    def initialize_fga(self, util_e, sharing_factor, flag, sizes, size_force=False, similar_modalities=None):
        return DummyFGA()

# mm_utils functionality inline
def separate_weights_from_bin(path, keyword):
    state_dict = torch.load(path, map_location="cpu")
    return {k.split(keyword + ".")[1]: v for k, v in state_dict.items() if keyword in k}

def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

# === Begin Test ===

model = DummyModel()

# Load mm_projector
if os.path.exists(pretrain_mm_mlp_adapter):
    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
    model.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
    print("? mm_projector weights loaded successfully.")
else:
    raise FileNotFoundError(f"mm_projector weights not found at {pretrain_mm_mlp_adapter}")

# Init FGA
num_of_patches = 2
sizes = [None] + [576 for _ in range(num_of_patches)]
text_dimension = model.config.hidden_size
vision_dimension = model.vision_tower.config.hidden_size
util_e = [text_dimension] + [vision_dimension for _ in range(num_of_patches)]
sharing_factor = {
    1: (1, [0]),
    2: (1, [0])
}
similar_modalities = [[i for i in range(2, num_of_patches + 1)]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compute_dtype = torch.float32  # or torch.bfloat16/float16 as needed
fga = model.initialize_fga(util_e, sharing_factor, False, sizes, size_force=False, similar_modalities=similar_modalities).to(dtype=compute_dtype, device=device)

# Load FGA weights
if os.path.exists(path_to_weights):
    print(f"Loading FGA weights from {path_to_weights}")
    weights = separate_weights_from_bin(path_to_weights, 'atten')
    fga.load_state_dict(weights, strict=False)
    print("? FGA weights loaded successfully.")
else:
    raise FileNotFoundError(f"FGA weights file not found at {path_to_weights}")
