import torch
import torch.nn as nn
import re
from .fga import fga

class fgabuilder:
    '''
    factor graph attention builder. 
    The input is a list of modlities and their hidden dimensions.
    The output is attended representations of each modality.
    '''
    def __init__(self, vocab_size, word_embed_dim, hidden_ques_dim, hidden_ans_dim,
                 hidden_hist_dim, hidden_cap_dim, hidden_img_dim):
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.hidden_ques_dim = hidden_ques_dim
        self.hidden_ans_dim = hidden_ans_dim
        self.hidden_hist_dim = hidden_hist_dim
        self.hidden_cap_dim = hidden_cap_dim
        self.hidden_img_dim = hidden_img_dim

    def build(self):
        return fga(self.vocab_size, self.word_embed_dim, self.hidden_ques_dim, self.hidden_ans_dim,
                   self.hidden_hist_dim, self.hidden_cap_dim, self.hidden_img_dim)
    
class MMAttn(nn.Module):
    def __init__(self, hidden_size, projector, num_heads=8, depth=1):
        super().__init__()
        self.hidden = hidden_size
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.projector = projector
        self.depth = depth
        self.inp_transform = nn.Linear(1024, hidden_size)

    def forward(self, x):
        for _ in range(self.depth):
            # Projector output as the query, and `x` as the key and value
            query = self.projector(x)
            print(x.size(), query.size(),self.hidden)
            x = self.inp_transform(x)
            attended_q = self.attn(x, query, query)[0]
            query  = query  + attended_q  # Residual connection to retain the original `x`
        return query


        
class PerceiverResampler(nn.Module):
    def __init__(self, hidden_size, num_queries=64, num_layers=2, from_pretrained=None):
        super(PerceiverResampler, self).__init__()
        self.num_layers = num_layers
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size))  # Learned queries
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        if from_pretrained:
            self.projector = from_pretrained
        else:
            self.projector = None

    def forward(self, x_f):
        if self.projector:
            x = self.projector(x_f)
        x = self.queries.expand(x_f.size(0), -1, -1)  # [Batch, R, d]
        for _ in range(self.num_layers):
            attn_output, _ = self.attn(x, torch.cat([x_f, x], dim=1), torch.cat([x_f, x], dim=1))
            x = x + attn_output
            x = x + self.ff(x)
        return x

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def get_projector_from_7b(config, ext=True, hidden_size=4096):
    # Define the projector structure
    mm_hidden_size = config.mm_hidden_size
    mlp_depth = 2
    modules = [nn.Linear(mm_hidden_size, hidden_size)]
    
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_size, hidden_size))
    if ext:
        # Add the new trainable layers
        modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_size, config.hidden_size))
        
    # Build the sequential model
    projector = nn.Sequential(*modules)
    return projector

def load_pretrained_projector(projector, strict=False, path='./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin'):
    # Load pretrained state dict
    pretrained_state_dict = torch.load(path, map_location="cpu")  # Adjust map_location as needed

    # Strip "model.mm_projector" prefix from keys if present
    updated_state_dict = {}
    for key, value in pretrained_state_dict.items():
        new_key = key.replace("model.mm_projector.", "")  # Remove prefix
        updated_state_dict[new_key] = value

    # Load the modified state dict into the projector
    projector.load_state_dict(updated_state_dict, strict=strict)  # Allow partial loading for new layers
    return projector

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
        
    if projector_type == 'from_pretrained':
        projector = load_pretrained_projector(get_projector_from_7b(config, ext=False), strict=True)
        return projector
    
    if projector_type == 'from_pretrained_ext':
        # Load the pretrained projector
        projector = load_pretrained_projector(get_projector_from_7b(config))
        # Freeze all parameters except the new layers
        for _, param in projector.named_parameters():
            # Freeze all parameters by default
            param.requires_grad = False

        # Set only the new layers to be trainable
        for _, param in list(projector.named_parameters())[-2:]:  # Last 4 entries (GELU + Linear + GELU + Linear)
            param.requires_grad = True

        return projector
        
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # NOTE: here you chnage the size of the input for the projection. What i did should fix layer 0.
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        #modules = [nn.Linear(1024, 4096)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            #modules.append(nn.Linear(4096, 4096))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    if projector_type == 'attn':
        return MMAttn(config.hidden_size, 
                      load_pretrained_projector(get_projector_from_7b(config, ext=False)), 
                      num_heads=kwargs.get('num_heads', 8), depth=kwargs.get('depth', 1))
    
    if projector_type == 'Perceiver_pretrained':
        # take the pretrained projector and add perciver to refine representation
        projector = load_pretrained_projector(get_projector_from_7b(config, ext=False), strict=True)
        return PerceiverResampler(config.hidden_size, num_layers=kwargs.get('num_layers', 2))
    
    raise ValueError(f'Unknown projector type: {projector_type}')
