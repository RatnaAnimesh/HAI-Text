import torch
from .generative_model import DiscretePOMDP

def forward_backward_update(model: DiscretePOMDP, observations, num_iterations=10, 
                            convergence_threshold=1e-4, 
                            custom_B_sem=None, custom_B_syn=None):
    """
    Bidirectional message passing (smoothing) for fixed sequence of observations.
    
    Computes:
    s_tau = softmax( ln A^T o_tau + ln B_{tau-1} s_{tau-1} + 
                     ln B_tau^T s_{tau+1} )
    
    Parameters:
    -----------
    model : DiscretePOMDP
    observations : list[int]
        Token indices of observed sequence
    num_iterations : int
        Number of fixed-point iterations
    convergence_threshold : float
        Stop if max change in beliefs is below this value
    custom_B_sem, custom_B_syn : torch.Tensor, optional
        Overrides for transition matrices (for hierarchical modulation)
    
    Returns:
    --------
    beliefs : list of (s_sem, s_syn) tuples
        Posterior belief at each time step
    """
    T = len(observations)
    
    # Determine which B matrices to use
    B_sem_use = custom_B_sem if custom_B_sem is not None else model.B_sem
    B_syn_use = custom_B_syn if custom_B_syn is not None else model.B_syn
    
    # Initialize beliefs to uniform (or from priors)
    beliefs = []
    s_sem, s_syn = model.initial_belief()
    for t in range(T):
        # We start with the initial (prior) belief for all timesteps or uniform.
        # Starting with the prior at t=0 and propagating roughly or just cloning is fine for initialization.
        beliefs.append((s_sem.clone(), s_syn.clone()))
    
    # Fixed-point iteration
    for iteration in range(num_iterations):
        beliefs_old = [(b[0].clone(), b[1].clone()) for b in beliefs]
        
        # We will compute updates for each time step t using messages from t-1 and t+1
        # Messages need to be recomputed based on current beliefs
        
        for t in range(T):
            # Likelihood term: log P(o_t | s_t)
            o_t = observations[t]
            log_likelihood = torch.log(model.A[o_t] + 1e-8)  # (n_sem, n_syn)
            
            # --- Forward Message (from t-1 to t) ---
            if t == 0:
                # At t=0, forward message is the prior D
                log_prior_fwd_sem = torch.log(model.D.sum(dim=1) + 1e-8)
                log_prior_fwd_syn = torch.log(model.D.sum(dim=0) + 1e-8)
            else:
                # Prior comes from t-1 transitioning to t. 
                # Action u_{t-1} is needed. In this passive perception phase, 
                # we assume the action taken was the token observed at t-1 (or some policy).
                # The user spec says: "Approximated action; use MAP or marginal. For now, assume action u = observed token"
                u_t_minus_1 = observations[t-1]
                
                # We use the previous BELIEF at t-1 to project forward
                s_sem_prev, s_syn_prev = beliefs[t-1]
                
                p_fwd_sem, p_fwd_syn = model.transition_prior(
                    s_sem_prev, s_syn_prev, u_t_minus_1,
                    B_sem=B_sem_use, B_syn=B_syn_use
                )
                log_prior_fwd_sem = torch.log(p_fwd_sem + 1e-8)
                log_prior_fwd_syn = torch.log(p_fwd_syn + 1e-8)
            
            # --- Backward Message (from t+1 to t) ---
            if t == T - 1:
                # No backward message from future
                log_prior_bwd_sem = torch.zeros_like(log_prior_fwd_sem)
                log_prior_bwd_syn = torch.zeros_like(log_prior_fwd_syn)
            else:
                # Backward message is trickier in standard formulation. 
                # It's usually: expected log P(s_{t+1} | s_t, u_t) evaluated at s_t.
                # However, the user spec essentially asks for a bidirectional approach where we check consistency.
                # "ln B_tau^T s_{tau+1}" suggests we look at how s_{t+1} projects BACKWARDS to s_t.
                # If B is stochastic matrix P(next|curr), B^T is roughly P(curr|next) scaled.
                # Let's implementation the intuition: what s_t caused s_{t+1}?
                
                u_t = observations[t] # Action taken at t
                s_sem_next, s_syn_next = beliefs[t+1]
                
                # We need ln( B.T @ s_next ). 
                # B_sem is (n_next, n_curr, vocab). 
                # B_sem[:, :, u_t] is (n_next, n_curr).
                # We want (n_curr,).
                # Transpose is (n_curr, n_next).
                # backward_msg = B^T @ s_next
                
                # Note: This is an approximation typical in Variational Message Passing.
                
                b_sem_T = B_sem_use[:, :, u_t].t() # (n_curr, n_next)
                b_syn_T = B_syn_use[:, :, u_t].t() # (n_curr, n_next)
                
                p_bwd_sem = b_sem_T @ s_sem_next
                p_bwd_syn = b_syn_T @ s_syn_next
                
                log_prior_bwd_sem = torch.log(p_bwd_sem + 1e-8)
                log_prior_bwd_syn = torch.log(p_bwd_syn + 1e-8)

            
            # --- Update Belief (Mean Field) ---
            # ln Q(s_sem) \propto E_{Q(syn)} [ ln P(o|s_sem, s_syn) ] + priors
            # E_{Q(syn)} [ ln A ] = sum_j Q(syn)_j * ln A[:, :, j]
            # This corresponds to weighted sum.
            
            # log_likelihood is (n_sem, n_syn)
            # s_syn is (n_syn)
            # We want (n_sem)
            # sum_j A_{ij} * s_j
            
            # Use current beliefs (from previous iteration) for the mean field expectation
            # We use `s_syn_prev` from beliefs[t] which is updated in place?
            # In the loop `for t in range(T)`, we use beliefs[t] at the start.
            # But we want the "other factor" belief.
            # We can use the one from the start of the 'iteration' loop (beliefs_old) for stability,
            # or the most recent one (beliefs[t]). Using 'beliefs[t]' is Gauss-Seidel-like.
            
            curr_s_sem, curr_s_syn = beliefs[t]
            
            # Update Sem using Syn
            # (n_sem, n_syn) * (1, n_syn) -> sum(1) -> (n_sem)
            likelihood_sem = (log_likelihood * curr_s_syn.unsqueeze(0)).sum(dim=1)
            
            log_joint_sem = likelihood_sem + log_prior_fwd_sem + log_prior_bwd_sem
            s_sem_new = torch.softmax(log_joint_sem, dim=0)
            
            # Update Syn using Sem (newly updated Sem? Or old? Standard VMP uses old for parallel or new for serial)
            # Let's use new Sem for faster convergence (Coordinate Ascent)
            # (n_sem, n_syn) * (n_sem, 1) -> sum(0) -> (n_syn)
            likelihood_syn = (log_likelihood * s_sem_new.unsqueeze(1)).sum(dim=0)
            
            log_joint_syn = likelihood_syn + log_prior_fwd_syn + log_prior_bwd_syn
            s_syn_new = torch.softmax(log_joint_syn, dim=0)
            
            beliefs[t] = (s_sem_new, s_syn_new)
        
        # Check convergence
        diffs = [torch.norm(b_new[0] - b_old[0]).item() + 
                 torch.norm(b_new[1] - b_old[1]).item()
                 for b_new, b_old in zip(beliefs, beliefs_old)]
        max_diff = max(diffs) if diffs else 0
        
        if max_diff < convergence_threshold:
            # print(f"Converged at iteration {iteration}")
            break
    
    return beliefs
