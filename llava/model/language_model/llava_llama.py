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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# ------------------- LLM Input Inspector -------------------
import random, torch, itertools, textwrap
from torch.nn import functional as F

class LLMInputLogger:
    def __init__(self, tokenizer, every_steps=200, sample=0):
        self.tokenizer   = tokenizer
        self.every_steps = every_steps
        self.sample      = sample
        self._step       = 0

    def __call__(self, input_ids, inputs_embeds, attention_mask,
                 position_ids, labels, logits=None):
        self._step += 1
        if self._step % self.every_steps:
            return

        s = self.sample  # which item in the batch to print
        emb_stats = dict(
            shape = list(inputs_embeds.shape),
            dtype = str(inputs_embeds.dtype),
            mu    = f"{inputs_embeds.mean():+.3e}",
            sigma = f"{inputs_embeds.std():.3e}",
            min   = f"{inputs_embeds.min():+.3e}",
            max   = f"{inputs_embeds.max():+.3e}",
        )

        print("\n" + "="*70)
        print(f"[LLM-LOGGER step {self._step}] inputs_embeds stats →", emb_stats)
        print(f"[LLM-LOGGER] attention_mask[{s}]  :", attention_mask[s, :50].tolist())
        if position_ids is not None:
            print(f"[LLM-LOGGER] position_ids[{s}]   :", position_ids[s, :50].tolist())

        # BOS-shift / padding / image-token check
        BOS = self.tokenizer.bos_token_id
        IMG = IMAGE_TOKEN_INDEX
        PAD = self.tokenizer.pad_token_id
        IGN = IGNORE_INDEX

        inp_txt = self.tokenizer.decode(
            [t for t in input_ids[s].tolist() if t not in (PAD, IMG)],
            skip_special_tokens=False
        )
        lbl_txt = self.tokenizer.decode(
            [t if t != IGN else PAD for t in labels[s].tolist()],
            skip_special_tokens=False
        )
        print("\n» INPUT  :", textwrap.shorten(inp_txt, 120))
        print("» LABEL  :", textwrap.shorten(lbl_txt, 120))

        if logits is not None:
            pred = logits[s, :-1].argmax(-1)        # teacher forcing: predict t+1
            pred_txt = self.tokenizer.decode(
                [t for t in pred.tolist() if t not in (PAD, IMG)],
                skip_special_tokens=False
            )
            print("» PRED   :", textwrap.shorten(pred_txt, 120))

        # quick invariants
        assert labels[s, 0] == IGN, "label[0] should be IGNORE (BOS shift)"
        assert (labels[s][input_ids[s] == IMG] == IGN).all(), "<image> token not masked"
        print("="*70 + "\n")

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    logger = None   # will hold the singleton


    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_image_features_and_multimodal(self, images):
        return self.encode_images(images), self.encode_images_no_proj(images)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if images is not None:
            image_sizes = [image.shape[-2:] for image in images]
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        # if LlavaLlamaForCausalLM.logger is None:
        #     LlavaLlamaForCausalLM.logger = LLMInputLogger(
        #         tokenizer=self.tokenizer, every_steps=200, sample=0
        #     )

        # # Optional: get logits too, so call the logger *after* fwd.
        # # We'll store inputs first
        # _log_inputs = (input_ids, inputs_embeds, attention_mask,
        #             position_ids, labels)
        # Forward pass through the main model
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # if LlavaLlamaForCausalLM.logger is not None:
        #     LlavaLlamaForCausalLM.logger(*_log_inputs, output.logits)

        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            image_sizes = [image.shape[-2:] for image in images]
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
