import numpy as np
import torch

def normalize_scores(scores):
    """Min-Max normalization"""
    min_val = scores.min(axis=1, keepdims=True)
    max_val = scores.max(axis=1, keepdims=True)
    return (scores - min_val) / (max_val - min_val + 1e-8)


def position_aware_list_reranking(llm_preference_scores, decay_rate, cfm_rating_mat, candidate_size):
    # position-aware list reranking
    sorted_indices = np.argsort(-llm_preference_scores, axis=1)
    llm_ranks = np.argsort(sorted_indices, axis=1)

    cfm_scores, cfm_idx = torch.topk(cfm_rating_mat, candidate_size, dim=1)
    normalized_cfm_scores = normalize_scores(cfm_scores.numpy())

    fusion_scores = (decay_rate ** llm_ranks) * normalized_cfm_scores
    final_reranks = (-fusion_scores).argsort(axis=1)

    final_reranked_item_list = np.array([[int(cfm_idx[u][x]) for x in row] for u, row in enumerate(final_reranks)])
    return final_reranked_item_list
