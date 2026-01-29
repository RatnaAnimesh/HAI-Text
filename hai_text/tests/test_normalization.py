import torch
from hai_text.models.validation import check_normalization
from hai_text.data.toy_grammar import create_random_pomdp

def test_random_pomdp_normalization():
    A, B_sem, B_syn, C, D = create_random_pomdp()
    
    valid_A, _ = check_normalization(A, dim=0)
    assert valid_A
    
    valid_B_sem, _ = check_normalization(B_sem, dim=0)
    assert valid_B_sem
    
    valid_B_syn, _ = check_normalization(B_syn, dim=0)
    assert valid_B_syn
    
    assert torch.allclose(D.sum(), torch.tensor(1.0))
