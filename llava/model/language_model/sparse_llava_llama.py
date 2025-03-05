from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .modelling_sparse_llama import LlamaDynamicvitModel, LlamaDynamicvitForCausalLM
from .llava_llama import LlavaLlamaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

class LlavaLlamaDynamicModel(LlavaMetaModel, LlamaDynamicvitModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaDynamicModel, self).__init__(config)

  
class LlavaLlamaDynamicForCausalLM(LlamaDynamicvitForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaDynamicvitForCausalLM.__init__(self,config)
        self.model = LlavaLlamaDynamicModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_shape = 576
        self.token_length_list = []
        self.pre_prompt_length_list = []
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        image_shape = 576,
        token_length_list = [],
        pre_prompt_length_list = [],
        retained_tokens = 192,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,          # None
                position_ids,       # None
                attention_mask,     # torch.Size([1, 668])
                past_key_values,    # None
                inputs_embeds,      # torch.Size([1, 668, 4096])
                labels,              # torch.Size([1, 668])
                image_shape,        
                token_length_list,  
                pre_prompt_length_list,     
            ) = self.prepare_sparse_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,                # None
            attention_mask=attention_mask,      # torch.Size([1, 668])
            position_ids=position_ids,          # None
            past_key_values=past_key_values,    # None
            inputs_embeds=inputs_embeds,        # torch.Size([1, 668, 4096])
            labels=labels,                      # torch.Size([1, 668])
            use_cache=use_cache,                # None
            output_attentions=output_attentions,# None
            output_hidden_states=output_hidden_states,  # None
            return_dict=return_dict,           # None
            image_shape = image_shape,
            token_length_list = token_length_list,
            pre_prompt_length_list = pre_prompt_length_list,
            retained_tokens=retained_tokens
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        retained_tokens = 192,
        image_shape=576,
        token_length_list = [],
        pre_prompt_length_list = [],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_shape,        
                token_length_list,  
                pre_prompt_length_list,     
            ) = self.prepare_sparse_inputs_labels_for_multimodal(       
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
            inputs_embeds=inputs_embeds,        # torch.Size([1, 664, 2048])
            image_shape = image_shape,        
            token_length_list = token_length_list,  
            pre_prompt_length_list = pre_prompt_length_list,     
            retained_tokens = retained_tokens,
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
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaDynamicForCausalLM)
