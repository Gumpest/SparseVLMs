'''
Copyright (2024) Peking University. 
Developers: Yuan Zhang, Chun-Kai Fan

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def attn_postprocess_rank(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, scale, bias):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start+v_token_num] # B, L2, L1

    rank = torch.linalg.matrix_rank(relation_vis_text.float()) # rank
    relation_vis_text = relation_vis_text.mean(1) # B, L1

    s_flag = True # layer needs sparsification or not
    if v_token_num - rank.item() != 0:
        mask = torch.zeros_like(relation_vis_text, dtype=bool)
        _, indices = torch.topk(relation_vis_text, min(int(rank.item() * scale + bias), v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False

    return mask, s_flag, relation_vis_text