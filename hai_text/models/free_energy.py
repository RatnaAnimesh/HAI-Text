import torch

def compute_free_energy(beliefs, observations, model):
    """
    F = KL[Q(s) || P(s)] - E_Q[ln P(o|s)]
      = Complexity - Accuracy
    
    Returns: (F_total, F_complexity, F_accuracy)
    """
    T = len(observations)
    
    # Complexity: KL between posterior and prior (at t=0)
    # NOTE: The provided spec snippet calculates KL only for t=0 ??
    # "kl_sem = torch.sum(beliefs[0][0] * ...)"
    # This implies we are treating the trajectory prior as factorized or just checking initialization?
    # Actually, full VFE for a trajectory Q(s_{0:T}) || P(s_{0:T}) splits into transition terms.
    # But the spec snippet is VERY specific:
    # "Complexity: KL between posterior and prior... s_sem_init, s_syn_init = model.initial_belief()"
    # It seems to calculate KL(Q(s_0) || P(s_0)). 
    # What about transitions P(s_t | s_{t-1})?
    # If the user wants EXACT implementation of the blueprint snippet, I will follow that.
    # The snippet:
    # kl_sem = torch.sum(beliefs[0][0] * (torch.log(beliefs[0][0]) - torch.log(s_sem_init)))
    
    # Wait, if this is a "Core Discrete POMDP" prototype, maybe we simplify Complexity to just initial state divergence 
    # OR we assume the "prior" in the loop handles the dynamics, and this FE function is a simplified metric?
    # I will implement EXACTLY what is in the snippet to satisfy "Focus on exact implementation".
    
    s_sem_init, s_syn_init = model.initial_belief()
    
    # Add epsilon to prevent log(0)
    kl_sem = torch.sum(beliefs[0][0] * 
                       (torch.log(beliefs[0][0] + 1e-8) - 
                        torch.log(s_sem_init + 1e-8)))
    kl_syn = torch.sum(beliefs[0][1] * 
                       (torch.log(beliefs[0][1] + 1e-8) - 
                        torch.log(s_syn_init + 1e-8)))
    F_complexity = kl_sem + kl_syn
    
    # Accuracy: expected log-likelihood
    F_accuracy = 0.0
    for t in range(T):
        o_t = observations[t]
        s_sem_t, s_syn_t = beliefs[t]
        # P(o|s)
        log_likelihood = torch.log(model.A[o_t] + 1e-8)  # (n_sem, n_syn)
        
        # E_Q[ln P(o|s)] = sum_{s_sem, s_syn} Q(s_sem) Q(s_syn) ln P(o|s_sem, s_syn)
        expected_ll = (s_sem_t.unsqueeze(1) * s_syn_t.unsqueeze(0) * 
                       log_likelihood).sum()
        F_accuracy += expected_ll
    
    # Note: Accuracy is usually subtracted in VFE (Energy - Entropy, or Complexity - Accuracy).
    # F = Complexity - Accuracy
    F_total = F_complexity - F_accuracy
    
    return F_total, F_complexity, F_accuracy
