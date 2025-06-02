import torch
import torch.nn as nn
import einops as ein
import math

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def cluster_and_merge(x,cluster_num):
    
    B, N, C = x.shape

    x1 = ein.rearrange(x, "b l r -> b l () r")
    x2 = ein.rearrange(x, "b l r -> b () l r")
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)        
    # get local density
    dist_nearest, index_nearest = torch.topk(dist_matrix, k=cluster_num, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    # add a little noise to ensure no tokens have the same density.
    density = density + torch.rand(
        density.shape, device=density.device, dtype=density.dtype) * 1e-6

    # get distance indicator
    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

    # select clustering center according to score
    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)    

    # assign tokens to the nearest center
    dist_matrix = index_points(dist_matrix, index_down)     

    idx_cluster = dist_matrix.argmin(dim=1)     

    # make sure cluster center merge to itself
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    # merge tokens

    B, N, C = x.shape
    device = dist_matrix.device
    idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
    agg_weight = x.new_ones(B, N, 1)
    
    token_weight = x.new_ones(B, N, 1)
    # self_attn_weights = self_attn_weights.mean(1)
    # token_weight = self_attn_weights.sum(dim=1).exp().unsqueeze(2) 
    # B_weight,N_weigh,C_weight = token_weight.shape
    # token_weight = token_weight.reshape(B_weight*N_weigh, C_weight)[sparse_token_idx.reshape(-1)].reshape(B, N, 1)
    
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num     

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),      
                            source=token_weight.reshape(B * N, 1))      
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]       

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),    
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)
    
    return x_merged
    
def  batch_index_select(x, idx):

    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        # in this condition
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def quick_batch_index_select(x, idx):

    B, N, C = x.size()
    N_new = idx.size(1)
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    idx = idx + offset
    out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
    
    return out

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_logits = attn_weight

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_logits