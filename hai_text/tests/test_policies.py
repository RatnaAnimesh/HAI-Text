import torch
from hai_text.models.generative_model import DiscretePOMDP
from hai_text.models.policies import evaluate_policy, sample_next_token_via_efe
from hai_text.data.toy_grammar import create_random_pomdp

def test_efe_pragmatic_value():
    """
    Test that policies aligning with preferences (C) have lower EFE.
    """
    vocab_size = 3
    # Setup simple model
    A, B_sem, B_syn, C, D = create_random_pomdp(vocab=vocab_size)
    
    # Preference: Token 0 is highly preferred, Token 1 is hated
    C[0] = 10.0
    C[1] = -10.0
    C[2] = 0.0
    
    model = DiscretePOMDP(A, B_sem, B_syn, C, D)
    current_belief = model.initial_belief()
    
    # Eval token 0 vs token 1
    # G = Pragmatic + Epistemic
    # Pragmatic = -C
    # G[0] should be ~ -10 (low EFE, good)
    # G[1] should be ~ +10 (high EFE, bad)
    
    G0 = evaluate_policy(model, [0], current_belief)
    G1 = evaluate_policy(model, [1], current_belief)
    
    # We can't guarantee epistemic part is identical, but pragmatic difference 
    # of 20 should dominate random epistemic differences in this toy setup.
    assert G0 < G1, f"Preferred token should have lower EFE. G0={G0}, G1={G1}"

def test_efe_epistemic_value():
    """
    Test that actions leading to uncertainty reduction are preferred.
    """
    # Harder to construct random test. 
    # Let's construct a deterministic transition case.
    # Action 0 -> leads to deterministic state (Low Entropy)
    # Action 1 -> leads to uniform state (High Entropy)
    
    A, B_sem, B_syn, C, D = create_random_pomdp(vocab=2, n_sem=2, n_syn=1)
    C = torch.zeros(2) # No preference
    
    # Start at state 0
    D = torch.zeros(2, 1); D[0, 0] = 1.0
    
    # B_sem:
    # Action 0: 0->0 deterministic
    # Action 1: 0->Uniform
    model = DiscretePOMDP(A, B_sem, B_syn, C, D)
    
    # Force B_sem logic
    # Shape: (n_to, n_from, vocab)
    model.B_sem[:, 0, 0] = torch.tensor([1.0, 0.0]) # Action 0 -> State 0
    model.B_sem[:, 0, 1] = torch.tensor([0.5, 0.5]) # Action 1 -> State 0/1 mix
    
    current_belief = (D.flatten(), torch.tensor([1.0])) # Sem, Syn
    
    # Entropy(State 0) = 0
    # Entropy(Uniform) > 0
    # Epistemic = -(H_before - H_after) = H_after - H_before
    # Action 0: H_after = 0. Epistemic = 0
    # Action 1: H_after > 0. Epistemic > 0
    # So Action 0 should have LOWER EFE (Minimize G).
    
    # Wait, "Epistemic value: expected information gain"
    # Usually we want High Info Gain.
    # Info Gain = H_prior - H_posterior (if we observe)
    # But here we are just predicting the FUTURE state H_next.
    # If we want to resolve uncertainty, we might pick actions that lead to "informative states"?
    # Actually, usually active inference seeks states that maximize Info Gain *about model parameters* or hidden states.
    # In this simplified "entropy reduction" formula provided by user:
    # epistemic = -(entropy_before - entropy_after)
    # We minimize G.
    # So we want (entropy_before - entropy_after) to be POSITIVE and LARGE.
    # => We want entropy_after < entropy_before.
    # We want to go to Low Entropy states.
    
    # In my setup:
    # Start: State 0 (Entropy 0).
    # Action 0: Next State 0 (Entropy 0). Delta = 0.
    # Action 1: Next State Uniform (Entropy High). Delta = negative.
    # So Action 0 is better (maintains low entropy). Action 1 increases entropy (bad).
    
    G0 = evaluate_policy(model, [0], current_belief)
    G1 = evaluate_policy(model, [1], current_belief)
    
    assert G0 < G1, f"Uncertainty reducing (or maintaining) action should be preferred. G0={G0}, G1={G1}"

def test_sampling():
    A, B_sem, B_syn, C, D = create_random_pomdp(vocab=5)
    C[0] = 100.0 # Super preferred
    model = DiscretePOMDP(A, B_sem, B_syn, C, D)
    
    candidates = [0, 1, 2, 3, 4]
    
    # Should almost always pick 0 due to softmax temperature
    # But it is stochastic.
    counts = {i:0 for i in range(5)}
    for _ in range(20):
        token, _ = sample_next_token_via_efe(model, model.initial_belief(), candidates, gamma=10.0)
        counts[token] += 1
        
    assert counts[0] > 10, "Should prefer token 0 significantly"
