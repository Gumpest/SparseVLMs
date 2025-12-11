import torch
import os

VERSION = os.getenv("USE_VERSION", "1_0")
V2_0 = VERSION == "2_0"

RETAIN_TOKN = int(os.getenv("RETAIN_TOKN", "192"))

layer_dict = {2:0,6:1,15:2}

sparse_token_list_192 = [300, 200, 110] if not V2_0 else [300, 200, 118]       # 2*576  4*300 10*200  16*110
sparse_token_list_128 = [303, 110, 36] if not V2_0 else [238, 108, 60]
sparse_token_list_96 = [238, 48, 26] if not V2_0 else [246, 54, 28]
sparse_token_list_64 = [66, 30, 17] if not V2_0 else [66, 34, 20]

sparse_token_dict = {
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    96 : sparse_token_list_96,
    64 : sparse_token_list_64
}

def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num] # B, L2, L1

    relation_vis_text = relation_vis_text.mean(1) # B, L1

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.

    sparse_token_list = sparse_token_dict[RETAIN_TOKN]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text

def select_attn_head_by_sum(self_attn_weights, t_token_idx, v_token_start, text_token_start):
    # [1,28,token_num,token_num] -> [28,text_token_num,visual_token_num]
    each_head_text_to_visual_attn = self_attn_weights[0][:, t_token_idx , v_token_start: text_token_start]
    # [28,text_token_num,visual_token_num] -> [28]
    sum_attn_per_head = each_head_text_to_visual_attn.sum((1,2))
    select_attn_head_idx = sum_attn_per_head.topk(14)[1]

    return self_attn_weights[:,select_attn_head_idx,:,:][:,:,:]

if __name__ == "__main__":

    self_attn_weights, v_token_start, v_token_num, text_token_start = torch.rand(4, 16, 1084, 1084), 36, 576, 700
    mask = attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start)
    print(mask.shape)