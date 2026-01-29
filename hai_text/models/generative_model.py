import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePOMDP(nn.Module):
    """
    Canonical POMDP: (O, S, U, P) where:
    - O: observation space (token indices, |O| = vocab_size)
    - S: state space (semantic × syntactic factors)
    - U: action space (token selection, |U| = vocab_size)
    - P: joint distribution over trajectories
    
    Now a learnable module with parameters stored as unnormalized logits.
    """
    
    def __init__(self, A=None, B_sem=None, B_syn=None, C=None, D=None,
                 vocab_size=None, n_sem=None, n_syn=None, frozen=False):
        """
        Initialize with either existing tensors (dense) or dimensions (random init).
        """
        super().__init__()
        
        # Determine shape
        if A is not None:
            vocab_size = A.shape[0]
            n_sem = A.shape[1]
            n_syn = A.shape[2]
        elif vocab_size is None or n_sem is None:
            raise ValueError("Must provide either tensors or dimensions (vocab_size, n_sem, n_syn)")
            
        self.vocab_size = vocab_size
        self.n_sem = n_sem
        self.n_syn = n_syn
        
        # Initialize Parameters (Logits)
        # Note: We use Parameter to enable gradient descent
        
        # A: (vocab, n_sem, n_syn)
        # We model P(o|s) = Softmax(A_logits, dim=0)
        self.A_logits = nn.Parameter(torch.randn(vocab_size, n_sem, n_syn))
        if A is not None:
            # Initialize from normalized A by taking log (approx)
            # Add small noise to avoid identical logits if A is uniform
            self.A_logits.data = torch.log(A + 1e-8)
            
        # B_sem: (n_sem, n_sem, vocab)
        self.B_sem_logits = nn.Parameter(torch.randn(n_sem, n_sem, vocab_size))
        if B_sem is not None:
            self.B_sem_logits.data = torch.log(B_sem + 1e-8)
            
        # B_syn: (n_syn, n_syn, vocab)
        self.B_syn_logits = nn.Parameter(torch.randn(n_syn, n_syn, vocab_size))
        if B_syn is not None:
            self.B_syn_logits.data = torch.log(B_syn + 1e-8)
            
        # D: (n_sem, n_syn)
        self.D_logits = nn.Parameter(torch.randn(n_sem, n_syn))
        if D is not None:
            self.D_logits.data = torch.log(D + 1e-8)
            
        # C: (vocab,) - Preferences are log-probs usually, or utility.
        # User defined C as "log P(o)_target". So C is NOT normalized usually.
        # We can treat C as a raw parameter.
        self.C_param = nn.Parameter(torch.zeros(vocab_size))
        if C is not None:
            self.C_param.data = C

        if frozen:
            for p in self.parameters():
                p.requires_grad = False
                
    @property
    def A(self):
        return F.softmax(self.A_logits, dim=0)
        
    @property
    def B_sem(self):
        return F.softmax(self.B_sem_logits, dim=0)
    
    @property
    def B_syn(self):
        return F.softmax(self.B_syn_logits, dim=0)
        
    @property
    def D(self):
        # D should sum to 1 over all states?
        # D is joint P(s_sem, s_syn). Flattens for softmax.
        shape = self.D_logits.shape
        start_flat = F.softmax(self.D_logits.view(-1), dim=0)
        return start_flat.view(shape)
        
    @property
    def C(self):
        return self.C_param

    def log_likelihood(self, obs_token, s_sem, s_syn):
        """
        Compute log P(o_t | s_t^sem, s_t^syn)
        """
        # Using A property which is softmaxed
        log_p = torch.log(self.A[obs_token] + 1e-8) 
        result = (s_sem.unsqueeze(1) * s_syn.unsqueeze(0) * log_p).sum()
        return result
    
    def transition_prior(self, s_sem_prev, s_syn_prev, action_token, B_sem=None, B_syn=None):
        """
        Compute P(s_t | s_{t-1}, u_{t-1})
        """
        B_sem_use = B_sem if B_sem is not None else self.B_sem
        B_syn_use = B_syn if B_syn is not None else self.B_syn
        
        p_s_sem_next = B_sem_use[:, :, action_token] @ s_sem_prev
        p_s_syn_next = B_syn_use[:, :, action_token] @ s_syn_prev
        
        return p_s_sem_next, p_s_syn_next
    
    def initial_belief(self):
        """
        Return P(s_1) as factorized (s_sem_dist, s_syn_dist)
        """
        d = self.D
        s_sem_dist = d.sum(dim=1)  # Marginal over syn
        s_syn_dist = d.sum(dim=0)  # Marginal over sem
        return s_sem_dist, s_syn_dist
