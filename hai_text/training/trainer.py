import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ..models.inference import forward_backward_update
from ..models.free_energy import compute_free_energy

class HAITrainer:
    def __init__(self, model, dataset, learning_rate=0.01, batch_size=4):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Determine if we can use DataLoader (requires collate if Variable length, 
        # but our dataset produces fixed seq_len)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    def train_epoch(self, epoch_idx=0):
        total_loss = 0.0
        total_batches = 0
        
        self.model.train() # Set to training mode (though dropout not used yet)
        
        for batch_idx, sequences in enumerate(self.loader):
            # sequences: (Batch, SeqLen)
            
            batch_loss = 0.0
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Since inference is not batched yet, loop over batch
            # and accumulate gradients (or accumulate loss then backward)
            # Accumulating loss is safer for autodiff graph
            
            for i in range(sequences.shape[0]):
                obs_seq = sequences[i].tolist()
                
                # 1. Run inference (forward-backward)
                # We need to ensure that 'beliefs' allows gradient flow?
                # The inference is a fixed point iteration logic.
                # In strict VFE learning ("Variational EM"), we optimize Q (beliefs) 
                # and P (params) alternatingly. 
                # E-step: finding Q (inference). This does not need Backprop through Q optimization usually?
                # Actually, standard VFE learning: 
                #   Minimize F(Q, theta) w.r.t Q (inference)
                #   Minimize F(Q, theta) w.r.t theta (M-step)
                # So we run inference to get 'beliefs' (Q*). Treating Q* as constant (detach),
                # we then compute F(Q*, theta) and backprop w.r.t theta.
                
                # Check if beliefs have grad? 
                # forward_backward_update creates new tensors. 
                # If we want to optimize Parameters A, B, we need gradients of F w.r.t A, B.
                # F depends on A via log_likelihood term.
                # F = Complexity - Accuracy.
                # Accuracy = E_Q [ln P(o|s)].
                # ln P(o|s) depends on A.
                # So even if Q is detached, F depends on theta.
                # So yes, we can detach beliefs.
                
                with torch.no_grad():
                    # E-Step: Infer beliefs (Q) using current parameters
                    # We don't want to backprop through the fixed-point loop iterations 
                    # into parameters, usually (unless doing meta-learning).
                    # Standard EM / Variational Bayes just treats Q as fixed for the update.
                    beliefs = forward_backward_update(
                        self.model, obs_seq, num_iterations=5
                    )
                
                # M-Step: Compute Free Energy F(Q, theta) and minimize
                # BUT, wait. 'compute_free_energy' calculates F.
                # Does `compute_free_energy` use `model.A`? Yes.
                # Does it adhere to the computation graph?
                # Yes, `log_likelihood = torch.log(model.A[o_t])`.
                # If beliefs are detached, gradient flows from F to model.A.
                
                # Note: compute_free_energy takes 'beliefs'. 
                # We assume beliefs are effectively constants here (from E-step).
                
                # Re-compute F *with* gradient enabled for parameters
                # We pass the detached 'beliefs' into it.
                
                fe, _, _ = compute_free_energy(beliefs, obs_seq, self.model)
                
                # Minimizing Free Energy maximizes Evidence
                # loss = FE
                
                # Normalize by sequence length? useful.
                loss = fe / len(obs_seq)
                batch_loss += loss
            
            # Average over batch
            batch_loss = batch_loss / sequences.shape[0]
            
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            total_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch_idx} | Batch {batch_idx} | Avg VFE: {batch_loss.item():.4f}")
                
        return total_loss / total_batches
