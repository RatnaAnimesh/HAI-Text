import torch
import random
from hai_text.models.generative_model import DiscretePOMDP
from hai_text.models.inference import forward_backward_update
from hai_text.data.toy_grammar import create_random_pomdp
from hai_text.models.validation import check_belief_validity

def test_belief_convergence():
    A, B_sem, B_syn, C, D = create_random_pomdp(vocab=20, n_sem=5, n_syn=5)
    model = DiscretePOMDP(A, B_sem, B_syn, C, D)
    obs = [random.randint(0, 19) for _ in range(5)]
    
    beliefs = forward_backward_update(model, obs, num_iterations=20, convergence_threshold=1e-5)
    
    # Check validity
    is_valid, msg = check_belief_validity(beliefs)
    assert is_valid, msg
    
    # Check if beliefs stabilized (run one more iter and compare)
    beliefs_next = forward_backward_update(model, obs, num_iterations=21, convergence_threshold=1e-5)
    
    for t in range(len(obs)):
        dist_sem = torch.norm(beliefs[t][0] - beliefs_next[t][0])
        dist_syn = torch.norm(beliefs[t][1] - beliefs_next[t][1])
        assert dist_sem < 1e-3
        assert dist_syn < 1e-3

def test_pomdp_initialization():
    A, B_sem, B_syn, C, D = create_random_pomdp()
    model = DiscretePOMDP(A, B_sem, B_syn, C, D)
    assert model.A.shape[0] == 200
    assert torch.allclose(model.A.sum(dim=0), torch.ones_like(model.A[0]))
