import torch
from hai_text.models.generative_model import DiscretePOMDP
from hai_text.models.hierarchical import HierarchicalPOMDP
from hai_text.data.toy_grammar import create_random_pomdp
from hai_text.models.inference import forward_backward_update

def test_hierarchical_coupling_shapes():
    """
    Verify that modulation dimensions work correctly.
    """
    # Low level model
    A, B_sem, B_syn, C, D = create_random_pomdp(vocab=10, n_sem=4, n_syn=3)
    pomdp_low = DiscretePOMDP(A, B_sem, B_syn, C, D)
    
    # High level model (context)
    # Vocab=1 (dummy), n_sem=2 (Context A vs Context B)
    A_h, B_sem_h, B_syn_h, C_h, D_h = create_random_pomdp(vocab=1, n_sem=2, n_syn=1)
    pomdp_high = DiscretePOMDP(A_h, B_sem_h, B_syn_h, C_h, D_h)
    
    # Coupling tensor: (n_high_sem, n_low_sem, n_low_sem, vocab)
    # 2 contexts, 4 low states
    coupling = torch.rand(2, 4, 4, 10)
    coupling = coupling / coupling.sum(dim=1, keepdim=True) # Normalize columns (dim 1)
    
    h_model = HierarchicalPOMDP(pomdp_low, pomdp_high, coupling)
    
    obs = [0, 1, 2]
    beliefs_low, beliefs_high = h_model.update_hierarchical(obs, num_iterations=2)
    
    assert len(beliefs_low) == 3
    # beliefs_high depends on how we summarize. Default is 1 step of high level.
    # Check shape
    assert beliefs_low[0][0].shape == (4,)
    
def test_top_down_modulation_effect():
    """
    Verify that changing the high-level prior changes the low-level posterior 
    for the exact same observation sequence.
    """
    vocab_size = 5
    n_low_sem = 2
    n_high_sem = 2
    
    # Low level
    A, B_sem, B_syn, C, D = create_random_pomdp(vocab=vocab_size, n_sem=n_low_sem, n_syn=2)
    pomdp_low = DiscretePOMDP(A, B_sem, B_syn, C, D)
    
    # High level
    A_h, B_sem_h, B_syn_h, C_h, _ = create_random_pomdp(vocab=1, n_sem=n_high_sem, n_syn=1)
    
    # Make distinct coupling matrices
    # Context 0: Strongly prefers staying in state 0 (Identity-like)
    B0 = torch.zeros(n_low_sem, n_low_sem, vocab_size)
    B0[0, 0, :] = 1.0
    B0[1, 1, :] = 1.0
    
    # Context 1: Strongly prefers swapping states (0->1, 1->0)
    B1 = torch.zeros(n_low_sem, n_low_sem, vocab_size)
    B1[1, 0, :] = 1.0
    B1[0, 1, :] = 1.0
    
    coupling = torch.stack([B0, B1]) # (2, 2, 2, 5)
    
    # CASE 1: High level prior is 100% Context 0
    D_h_0 = torch.tensor([[1.0], [0.0]]) # n_sem=2, n_syn=1 (flattened or shaped?)
    # D is (n_sem, n_syn). D_h shape is (2, 1)
    D_h_0 = torch.zeros(2, 1); D_h_0[0, 0] = 1.0
    pomdp_high_0 = DiscretePOMDP(A_h, B_sem_h, B_syn_h, C_h, D_h_0)
    
    h_model_0 = HierarchicalPOMDP(pomdp_low, pomdp_high_0, coupling)
    obs = [0, 0, 0] # Some obs
    
    # Run
    beliefs_low_0, _ = h_model_0.update_hierarchical(obs)
    
    # CASE 2: High level prior is 100% Context 1
    D_h_1 = torch.zeros(2, 1); D_h_1[1, 0] = 1.0
    pomdp_high_1 = DiscretePOMDP(A_h, B_sem_h, B_syn_h, C_h, D_h_1)
    
    h_model_1 = HierarchicalPOMDP(pomdp_low, pomdp_high_1, coupling)
    beliefs_low_1, _ = h_model_1.update_hierarchical(obs)
    
    # The low level transition dynamics are different, so the inferred beliefs should differ
    # We check if beliefs at t=1 or t=2 are different
    
    # Distance between belief trajectories
    diff = 0.0
    for t in range(len(obs)):
        diff += torch.norm(beliefs_low_0[t][0] - beliefs_low_1[t][0]).item()
        
    print(f"Difference between context 0 and 1: {diff}")
    assert diff > 1e-4, "Top-down context should alter low-level beliefs"
