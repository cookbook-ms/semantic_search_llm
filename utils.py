import numpy as np

def compute_ndcg(rank_indices, scores, k=None):
    """
    Compute NDCG@k given reranked item indices and their scores.
    
    Args:
        rank_indices (list of int): List of indices in the reranked order (1-based).
        scores (dict): Dictionary mapping index (0-based) to score.
        k (int or None): Number of top results to evaluate. Defaults to len(rank_indices).

    Returns:
        float: NDCG@k value
    """
    if k is None:
        k = len(rank_indices)
        
    # Step 1: Get scores in predicted order
    # predicted_scores = [scores[idx] for idx in rank_indices[:k]]
    predicted_scores = scores 

    # Step 2: Compute DCG
    dcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(predicted_scores))

    # Step 3: Compute ideal DCG (sorted scores)
    ideal_scores = sorted(predicted_scores, reverse=True)
    idcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(ideal_scores))

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    return dcg / idcg