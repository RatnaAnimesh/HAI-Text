import torch

def entropy(distribution):
    """
    Compute Shannon entropy of a distribution.
    H(s) = - sum s * ln(s)
    """
    # Add epsilon to avoid log(0)
    return -torch.sum(distribution * torch.log(distribution + 1e-8))

def evaluate_policy(model, policy_tokens, current_belief):
    """
    Compute Expected Free Energy for a sequence of future actions.
    
    G(π) ≈ pragmatic_value + epistemic_value
    
    Parameters:
    -----------
    policy_tokens : list[int]
        Sequence of future actions (tokens)
    current_belief : (s_sem, s_syn)
    
    Returns:
    --------
    G : float
        EFE (lower is better, assuming G is minimized EFE)
        Note: EFE definition varies. 
        G ≈ - Expected Log Preference - Expected Info Gain
        So minimizing G maximizes preference and info gain.
    """
    G = 0.0
    s_sem, s_syn = current_belief
    
    # We clone to avoid modifying the input belief tensors during simulation
    s_sem = s_sem.clone()
    s_syn = s_syn.clone()
    
    for t, u_t in enumerate(policy_tokens):
        # Predicted next state Q(s_{t+1}|s_t, \pi)
        # Using transition prior
        s_sem_next, s_syn_next = model.transition_prior(s_sem, s_syn, u_t)
        
        # --- Pragmatic value: Risk / Divergence from Preference ---
        # Often defined as D_KL(Q(o) || P(o)). 
        # Approximated as: E_Q[ln P(o)_taget]
        # model.C[u_t] is log P(o)_target.
        # But here we are selecting actions u_t directly. The action generates the observation deterministically 
        # (in this active generation context where "action" = "output token").
        # So we just prefer actions that match C.
        # "Pragmatic value: how well does this action align with preferences?"
        # G_pragmatic = - ln P(u_t) = -C[u_t]
        
        pragmatic = -model.C[u_t]  # Negative because C is log-preference, we minimize G
        
        # --- Epistemic value: Ambiguity or Expected Information Gain ---
        # G_epistemic = - E_Q[D_KL(Q(s|o) || Q(s))] 
        # Approx: H(Q(s)) - E_Q[H(Q(s|o))]
        # The user provided approximation: "info_gain ≈ entropy_before - entropy_after"
        # This is a bit non-standard (usually it's MI), but let's stick to the user's specific formula:
        # "epistemic = -(entropy_before - entropy_after)"
        # This implies we want entropy to DECREASE (high info gain).
        
        entropy_before = entropy(s_sem) + entropy(s_syn)
        entropy_after = entropy(s_sem_next) + entropy(s_syn_next)
        
        # If entropy drops (high uncertainty -> low uncertainty), (before - after) is positive.
        # We want to maximize this reduction.
        # So we minimize negative reduction.
        epistemic = -(entropy_before - entropy_after)
        
        G += pragmatic + epistemic
        
        # Update belief for next step in horizon
        s_sem, s_syn = s_sem_next, s_syn_next
    
    return G

def sample_next_token_via_efe(model, current_belief, candidate_tokens, 
                              horizon=1, gamma=1.0):
    """
    Select next token by evaluating EFE over candidate actions.
    
    P(u) ∝ exp(-γ G(u))
    """
    efe_scores = []
    
    # Evaluate each candidate action
    for u_cand in candidate_tokens:
        # Simple lookahead
        # We could perform tree search, but for now just 1-step or rollout
        efe = evaluate_policy(model, [u_cand], current_belief)
        efe_scores.append(efe)
    
    efe_scores = torch.tensor(efe_scores)
    
    # Softmax selection (Minimizing G -> Maximizing -G)
    probs = torch.softmax(-gamma * efe_scores, dim=0)
    
    # Use torch.multinomial for weighted sampling
    sample_idx = torch.multinomial(probs, 1).item()
    next_token = candidate_tokens[sample_idx]
    
    return next_token, efe_scores
