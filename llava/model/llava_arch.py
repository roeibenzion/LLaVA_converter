#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn.functional as F
from abc import ABC, abstractmethod
from .multimodal_projector.fga.attn import Atten 

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

class CrossAttentionLayer(nn.Module):
    def __init__(self, llm_hidden_size, image_hidden_size, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.llm_hidden_size = llm_hidden_size
        self.image_hidden_size = image_hidden_size

        # Projection layers to align dimensions
        self.key_proj = nn.Linear(image_hidden_size, llm_hidden_size)
        self.value_proj = nn.Linear(image_hidden_size, llm_hidden_size)

        # Multi-head attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llm_hidden_size,  # Attention mechanism operates in the LLM space
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Inputs are (batch_size, seq_len, embed_dim)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(llm_hidden_size)

        # Optional feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size * 4, llm_hidden_size),
        )
        self.ff_norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, llm_features, image_features):
        """
        Perform cross-attention between LLM features and image features.

        Args:
            llm_features (torch.Tensor): [batch_size, seq_len, llm_hidden_size]
            image_features (torch.Tensor): [batch_size, num_patches, image_hidden_size]

        Returns:
            torch.Tensor: Updated LLM features after cross-attention.
        """
        # Project image features to match llm_hidden_size
        key = self.key_proj(image_features)  # [batch_size, num_patches, llm_hidden_size]
        value = self.value_proj(image_features)  # [batch_size, num_patches, llm_hidden_size]

        # Compute attention
        attn_output, _ = self.cross_attention(
            query=llm_features,  # [batch_size, seq_len, llm_hidden_size]
            key=key,             # [batch_size, num_patches, llm_hidden_size]
            value=value          # [batch_size, num_patches, llm_hidden_size]
        )

        # Add & Normalize
        llm_features = self.norm(llm_features + attn_output)
        # Feed-forward
        ff_output = self.feed_forward(llm_features)
        llm_features = self.ff_norm(llm_features + ff_output)

        return llm_features


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    print(original_size)
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_images_no_proj(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        return image_features

    def initialize_fga(self, util_e, sharing_factor,prior_flag, sizes, size_force):
        self.atten = Atten(util_e, sharing_factor, prior_flag, sizes, size_force)
        return self.atten
    
    def get_visual_tokens(self, images, image_features, image_sizes = None):
        #split_sizes = [image.shape[0] for image in images]
        split_sizes = images.shape[1]
        image_features = torch.split(image_features, split_sizes, dim=0)
        #mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        mm_patch_merge_type = 'flat'
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        if mm_patch_merge_type == 'flat':
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith('spatial'):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == 'anyres':
                        '''
                        try:num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                        except Exception as e:
                            print(f"Anyres Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        '''
                        num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        raise NotImplementedError
                    if 'unpad' in mm_patch_merge_type:
                        print("before first unpad: ", image_feature.shape)
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        print("after first unpad: ", image_feature.shape)
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        print("after second unpad: ", image_feature.shape)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        print("after third unpad: ", image_feature.shape)
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                        ), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                        print(image_feature.shape)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[None].to(image_feature.device)
                        ), dim=0)
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        return image_features
    def inputs_for_atten(self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None):
        print("1")
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                num_patches_per_image = [image.shape[0] for image in images]
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            else:
                num_patches_per_image = [image.shape[0] for image in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images_no_proj(concat_images)
            print("2")
        else:
            image_features = self.encode_images_no_proj(images)
        X_v = image_features
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        H_q = []
        num_images_per_batch = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                H_q.append(cur_input_embeds_1)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            H_q.append(cur_input_embeds_no_im[-1])
            num_images_per_batch.append(num_images)
        # pad the text tokens to the same length
        max_len = 50
        for i in range(len(H_q)):
            #NOTE: size of embedding should be the same for all samples. They did it with 21, i'll do it with more.
            H_q[i] = F.pad(H_q[i], (0, 0, 0, max_len - H_q[i].size(0)), value=0)
        H_q = torch.stack(H_q)
        # iterate over the image patches
        split_images = torch.split(X_v, num_patches_per_image, dim=0)
        '''
        new_X_v = []
        for batch_idx, image in enumerate(split_images):
            patches_atten = []
            for i in range(image.shape[0]):
                image_patch = image[i]
                rest_of_patches_list = [image[j].unsqueeze(0) for j in range(image.shape[0]) if j != i]
                params = [H_q[batch_idx].unsqueeze(0), image_patch.unsqueeze(0)] + rest_of_patches_list
                patches_atten.append(self.atten(params)[1])
            new_X_v.append(torch.stack(patches_atten, dim=1))
        X_v = torch.cat(new_X_v, dim=0)
        '''
        # num of patches
        N = split_images[0].shape[0]
        patches_atten = []

        for i in range(N):
            center_patch = torch.stack([split_images[j][i] for j in range(len(split_images))])
            rest_patches = [torch.stack([split_images[j][k] for j in range(len(split_images))]) 
                            for k in range(N) if k != i]
            params = [H_q, center_patch] + rest_patches
            print(len(params))
            out = self.atten(params)[1]
            patches_atten.append(out)

        X_v= torch.stack(patches_atten, dim=1)
        
        print(f"the size of X_v: {X_v.shape}")
        # project 
        image_features = self.get_model().mm_projector(X_v)
        print(f"the size of image_features: {image_features.shape}")
        image_features = self.get_visual_tokens(images, image_features, image_sizes)
        print(f"the size of image_features #2: {len(image_features)}")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = num_images_per_batch[batch_idx]
            cur_new_input_embeds = []
            cur_new_labels = []
            
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        use_attn = hasattr(self, "atten") and self.atten is not None
        if use_attn:
            print("using attention")
            return self.inputs_for_atten(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            if use_attn:
                print(f"before encode images {concat_images.shape}")
                image_features = self.encode_images_no_proj(concat_images)
            else:
                image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            print(f"image features: {len(image_features)}")
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                X_v = torch.stack(image_features)
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        print("image feature shape: ", image_feature.shape)
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            '''
                            try:num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            except Exception as e:
                                print(f"Anyres Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            '''
                            num_patch_width, num_patch_height = 2, 2
                            print(f"before reshape: {image_feature.shape}")
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            print(f"after reshape: {image_feature.shape}")
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            print("before first unpad: ", image_feature.shape)
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            print("after first unpad: ", image_feature.shape)
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            print("after second unpad: ", image_feature.shape)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            print("after third unpad: ", image_feature.shape)
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            print(image_feature.shape)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
                X_v = torch.stack(image_features)
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
             image_features = self.encode_images(images)
             X_v = self.encode_images_no_proj(images)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        H_q = []
        num_images_per_batch = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                H_q.append(cur_input_embeds_1)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            H_q.append(cur_input_embeds_no_im[-1])
            num_images_per_batch.append(num_images)
            if use_attn:
                continue
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            print("here is size and len summery #1: ")
            print(f"new input embeds: {len(new_input_embeds)}")
            print(f"new labels: {len(new_labels)}")
            print(f"new input embeds[0]: {new_input_embeds[0].shape}")
            print(f"new labels[0]: {new_labels[0].shape}")
        
        if use_attn:
            # pad the text tokens to the same length
            max_len = 50
            for i in range(len(H_q)):
                #NOTE: size of embedding should be the same for all samples. They did it with 21, i'll do it with more.
                H_q[i] = F.pad(H_q[i], (0, 0, 0, max_len - H_q[i].size(0)), value=0)
            H_q = torch.stack(H_q)
            # send H_q and X_v to the attention module
            X_v = self.atten([H_q, X_v])[1]
            X_v = X_v.to(self.get_model().mm_projector[0].weight.dtype)
            # Get it to a one token represtation so [bs, 1, text_dim]
            X_v = X_v.unsqueeze(1)
            # project 
            image_features = self.get_model().mm_projector(X_v)
            # compensate for the loop in the previous code
            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = num_images_per_batch[batch_idx]
                cur_new_input_embeds = []
                cur_new_labels = []
                
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
                print("here is size and len summery #2: ")
                print(f"new input embeds: {len(new_input_embeds)}")
                print(f"new labels: {len(new_labels)}")
                print(f"new input embeds[0]: {new_input_embeds[0].shape}")
                print(f"new labels[0]: {new_labels[0].shape}")

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
