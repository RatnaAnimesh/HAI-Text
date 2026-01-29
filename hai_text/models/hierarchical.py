import torch
from .generative_model import DiscretePOMDP
from .inference import forward_backward_update

class HierarchicalPOMDP:
    """
    Two-level hierarchy:
    - Level 1 (low): token/lexical level
    - Level 2 (high): phrase/intent level
    
    Link function (Section 4.2.2):
    B^(1) = sum_k s_k^(2) * B_k^(1)
    """
    
    def __init__(self, pomdp_low: DiscretePOMDP, pomdp_high: DiscretePOMDP, 
                 coupling_B_sem):
        """
        Parameters:
        -----------
        pomdp_low : DiscretePOMDP
            The token-level model.
        pomdp_high : DiscretePOMDP
            The intent-level model.
        coupling_B_sem : torch.Tensor
            Tensor of shape (n_high_sem, n_low_sem, n_low_sem, vocab).
            coupling_B_sem[k] is the transition matrix B_sem for the low level 
            when high level semantic state is k.
        """
        self.pomdp_low = pomdp_low
        self.pomdp_high = pomdp_high
        self.coupling_B_sem = coupling_B_sem 
        # For simplicity in this prototype, we only couple Semantic states.
        # Syntactic transitions might be fixed or also coupled, but let's start with Semantic coupling.
        
        # Validation
        assert coupling_B_sem.shape[0] == pomdp_high.D.shape[0] # n_high_sem
        assert coupling_B_sem.shape[1] == pomdp_low.D.shape[0]  # n_low_sem
        
    def _modulate_transitions(self, s_high_sem):
        """
        Compute B_sem_modulated = sum_k s_high_sem[k] * coupling_B_sem[k]
        """
        # s_high_sem: (n_high_sem,)
        # coupling_B_sem: (n_high_sem, n_low_sem, n_low_sem, vocab)
        
        # Einsum: k, kijv -> ijv
        B_modulated = torch.einsum('k,kijv->ijv', s_high_sem, self.coupling_B_sem)
        
        # Re-normalize just in case numerical errors drift (optional but safe)
        # B_modulated = B_modulated / B_modulated.sum(dim=0, keepdim=True)
        return B_modulated
    
    def update_hierarchical(self, observations, high_level_obs_fn=None, num_iterations=5):
        """
        Alternated inference: high-level prediction -> low-level inference
        
        Parameters:
        -----------
        observations : list[int]
            Low-level token indices.
        high_level_obs_fn : callable
             Function mapping low-level obs to high-level obs. 
             If None, uses a dummy [0] observation.
        """
        
        # 1. Upward: infer high-level states
        # For a static high-level Context, we treat the whole sequence as one "step" for the high level?
        # Or does the high level evolve?
        # In the "Deep Temporal Model", high level evolves slower. 
        # For this "Phase 2" simple coupling, let's assume high level is a SINGLE state (T=1) static context
        # that generates the sequence.
        
        if high_level_obs_fn is None:
            high_obs = [0] # Dummy
        else:
            high_obs = high_level_obs_fn(observations)
            
        beliefs_high = forward_backward_update(self.pomdp_high, high_obs, num_iterations=num_iterations)
        
        # 2. Modulate low-level dynamics
        # We take the belief at t=0 (or average) of the high level
        s_high_sem = beliefs_high[0][0] # (n_high_sem,)
        
        B_sem_modulated = self._modulate_transitions(s_high_sem)
        
        # 3. Downward: infer low-level states with modulated B
        beliefs_low = forward_backward_update(
            self.pomdp_low, observations, 
            num_iterations=num_iterations,
            custom_B_sem=B_sem_modulated
            # custom_B_syn left as default (None -> model.B_syn)
        )
        
        return beliefs_low, beliefs_high
