import torch

def check_normalization(tensor, dim=0, tolerance=1e-5):
    """
    Check if a tensor is normalized along a dimension.
    """
    sums = tensor.sum(dim=dim)
    is_valid = torch.allclose(sums, torch.ones_like(sums), atol=tolerance)
    return is_valid, sums

def check_belief_validity(beliefs):
    """
    Check if a list of (s_sem, s_syn) beliefs are valid distributions.
    """
    for t, (s_sem, s_syn) in enumerate(beliefs):
        if not torch.allclose(s_sem.sum(), torch.tensor(1.0), atol=1e-4):
            return False, f"s_sem at t={t} not normalized"
        if not torch.allclose(s_syn.sum(), torch.tensor(1.0), atol=1e-4):
            return False, f"s_syn at t={t} not normalized"
        if (s_sem < 0).any() or (s_syn < 0).any():
            return False, f"Negative probabilities at t={t}"
    return True, "Valid"
