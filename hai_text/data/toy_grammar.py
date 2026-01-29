import torch
import random
from ..config import VOCAB_SIZE, N_SEMANTIC_STATES, N_SYNTACTIC_STATES

def create_random_pomdp(vocab=VOCAB_SIZE, n_sem=N_SEMANTIC_STATES, n_syn=N_SYNTACTIC_STATES):
    """
    Generate random normalized probability matrices for testing
    """
    # A: P(o | s_sem, s_syn)
    A = torch.rand(vocab, n_sem, n_syn)
    A = A / A.sum(dim=0, keepdim=True)
    
    # B_sem: P(s'_sem | s_sem, u)
    B_sem = torch.rand(n_sem, n_sem, vocab)
    B_sem = B_sem / B_sem.sum(dim=0, keepdim=True)
    
    # B_syn: P(s'_syn | s_syn, u)
    B_syn = torch.rand(n_syn, n_syn, vocab)
    B_syn = B_syn / B_syn.sum(dim=0, keepdim=True)
    
    # C: Preferences (log probs)
    C = torch.randn(vocab)
    
    # D: Initial belief
    D = torch.rand(n_sem, n_syn)
    D = D / D.sum()
    
    return A, B_sem, B_syn, C, D

def get_toy_grammar():
    """
    Returns vocabulary and simple transition logic instructions if needed.
    For this prototype, we rely on the random matrices for unit tests
    and will build a specific 'cat eats fish' matrix set for the validaton script.
    """
    # Basic vocabulary
    vocab = ["cat", "dog", "fish", "eats", "sees", "chases", ".", "the", "a"]
    return vocab
