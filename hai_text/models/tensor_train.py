import torch
import torch.nn as nn
import torch.nn.functional as F

class TTMatrix(nn.Module):
    """
    Represents a Matrix Product Operator (MPO) / Tensor Train Matrix.
    Used for efficient storage and multiplication of large matrices (e.g., transition matrices B).
    
    Shape: (d_out_total, d_in_total) factored into d cores.
    Each core has shape: (rank_in, dim_out, dim_in, rank_out)
    """
    def __init__(self, in_dims, out_dims, ranks):
        """
        Args:
            in_dims: List of input dimensions [d_in_1, d_in_2, ...]
            out_dims: List of output dimensions [d_out_1, d_out_2, ...]
            ranks: List of connection ranks [r_0=1, r_1, ..., r_d=1]
                   Length must be len(in_dims) + 1.
        """
        super().__init__()
        assert len(in_dims) == len(out_dims)
        assert len(ranks) == len(in_dims) + 1
        assert ranks[0] == 1 and ranks[-1] == 1
        
        self.cores = nn.ParameterList()
        self.num_cores = len(in_dims)
        
        for i in range(self.num_cores):
            # Shape: (r_in, d_out, d_in, r_out)
            shape = (ranks[i], out_dims[i], in_dims[i], ranks[i+1])
            # Initialize with random normal scaled by fan_in
            core = torch.randn(shape) / (ranks[i] * in_dims[i])**0.5
            self.cores.append(nn.Parameter(core))
            
    def metric_projection(self, vector):
        """
        Multiplies the TT-matrix by a dense vector.
        y = M * x
        
        This corresponds to 'contracting' the vector with the operator.
        
        Args:
           vector: Dense tensor of shape (batch, d_in_1, d_in_2, ...) 
                   or (d_in_1, d_in_2, ...)
        
        Returns:
           Result tensor of shape (batch, d_out_1, ...) or (d_out_1, ...)
        """
        # Handle batch dimension if present
        has_batch = vector.dim() > self.num_cores
        if not has_batch:
            vector = vector.unsqueeze(0)
            
        bs = vector.shape[0]
        original_bs = bs
        # Initial rank tensor: shape (batch, 1)
        rank_tensor = torch.ones(bs, 1, device=vector.device)
        
        # Contract cores from left to right
        # Core: (r_in, d_out, d_in, r_out)
        # Current accumulated tensor shape: (batch, r_in) 
        # We need to act on the vector dimensions one by one
        
        # NOTE: The standard way to multiply MPO x Vector is to contract site by site.
        # Temp tensor shape starts at (batch, r_0=1)
        curr = rank_tensor
        
        # Vector shape: (batch, d_in_1, d_in_2, ...)
        # We reshape vector to access d_in easily if needed, but it's already structured
        
        for i, core in enumerate(self.cores):
            # core: (r_in, d_out, d_in, r_out)
            # curr: (batch, r_in, partially_contracted_d_out...) - Wait, standard sweep is better
            
            # Let's do it sequentially.
            # curr represents the contraction of the left machinery.
            # shape: (batch, r_in) -- NO, this is for scalar output.
            
            # For Matrix-Vector multiplication:
            # We are mapping In -> Out.
            # We contract the 'd_in' index of the core with the 'd_in' index of the vector.
            # But the vector is fully entangled (dense).
            # So we usually contract one index of the vector at a time.
            
            # Let's maintain a tensor 'T' of shape (batch, r_current, remaining_vector_dims...)
            # Actually, standard algorithm:
            # T_0 = vector (batch, d_in_1, d_in_2, ...)
            # Contract T_0 with Core_1 on d_in_1:
            #   T_0 shape: (batch, d_in_1, d_in_2, ...)
            #   Core_1 shape: (r_0, d_out_1, d_in_1, r_1)
            #   Result T_1: (batch, d_in_2, ..., r_0, d_out_1, r_1) -> reshape/permute
            
            # Improved loop:
            # We accumulate the *output* dimensions and carry the rank.
            pass
        
        # Let's retry the loop logic for MPO x Dense Vector -> Dense Vector
        # We can treat the dense vector as a tensor with shape (batch, d_in_1, ..., d_in_N)
        # We contract dimension d_in_i at step i.
        
        res = vector
        # res shape: (batch, d_in_1, ..., d_in_N)
        
        # We need to introduce the rank dimension initially? No.
        
        # This is surprisingly tricky to write generically for N dimensions in PyTorch 
        # without einsum over variable strings.
        # Let's assume we proceed dimension by dimension.
        
        # Work with shape: (batch, r_prev, d_in_i, remaining_ins...) -> (batch, r_next, d_out_i, remaining_ins...)
        
        # Initial setup: Treat (batch) as (batch, r_0=1)
        res = res.unsqueeze(1) # (batch, 1, d_in_1, ...)
        
        collected_outs = []
        
        for i in range(self.num_cores):
            # core: (r_in, d_out, d_in, r_out)
            # res:  (batch, r_in, d_in_i, d_in_{i+1}...)
            
            # We want to contract along r_in and d_in_i
            # And produce (batch, r_out, d_out_i, d_in_{i+1}...)
            
            d_in = self.cores[i].shape[2]
            
            # Permute res to bring d_in_i to the front for contraction
            # res is (batch, r_in, d_in_i, rest...)
            # It's already in position if we just index correctly.
            
            # Flatten rest:
            # res: (batch, r_in, d_in, rest)

            # Core:        (r_in, d_out, d_in_current, r_out)
            
            # We want:     (batch, r_out, d_out, d_in_future...)
            
            # Let's isolate d_in_current
            # res shape is (batch, r_in, d_in_i, d_in_{i+1}, ..., d_in_N)
            
            # Pop d_in_i.
            # reshape res to (batch, r_in, d_in_i, -1)
            remaining_shape = res.shape[3:]
            res_temp = res.reshape(bs, self.cores[i].shape[0], self.cores[i].shape[2], -1) 
            # (batch, r_in, d_in, rest)
            
            # Einsum:
            # b: batch
            # r: r_in
            # i: d_in
            # x: rest
            # o: d_out
            # k: r_out
            # T = core(r, o, i, k)
            # V = res(b, r, i, x)
            # Out = (b, k, o, x)
            
            new_res = torch.einsum('roik,brix->bkox', self.cores[i], res_temp)
            
            # new_res: (batch, r_out, d_out, rest)
            # We want to "store" d_out and keep processing "rest".
            # Structurally, MPO usually produces an output tensor structure.
            # Let's accumulate d_out to the LEFT or maintain it?
            
            # If we keep d_out in the tensor, the tensor grows:
            # Step 1: (batch, r1, d_out1, d_in2...)
            # Step 2: (batch, r2, d_out2, d_out1, d_in3...) -> this gets messy with order
            
            # Better approach:
            # Keep `outputs` aside.
            # But the ranks entangle them.
            
            # Standard MPS contraction strategies:
            # If we just want the result vector, we are mapping (d_in_1...d_in_N) to (d_out_1...d_out_N).
            # The result is a dense tensor.
            
            # Let's push d_out to the "batch" side temporarily? No.
            # Let's keep it simple.
            # new_res shape: (batch, r_out, d_out, rest_of_d_ins)
            
            # Move d_out to a 'completed' list? No, it's entangled.
            # We just need to ensure the next iteration sees (batch, r_in', d_in_{i+1}...)
            # So we permute d_out to the left-most (after batch) or combine with batch?
            # If we combine with batch, the batch size grows! (batch * d_out).
            # Yes, that effectively treats the output dimensions as "batch" for the subsequent cores.
            
            # new_res: (batch, r_out, d_out, rest)
            # permute to (batch, d_out, r_out, rest)
            # reshape to (batch * d_out, r_out, rest...)
            
            new_res = new_res.permute(0, 2, 1, 3).reshape(bs * self.cores[i].shape[1], self.cores[i].shape[3], *remaining_shape)
            
            # Update batch size for next iteration
            bs = bs * self.cores[i].shape[1]
            res = new_res
        
        # End loop
        # res shape: (batch * d_out_1 * ... * d_out_N, r_last=1)
        res = res.squeeze(-1) # Remove rank dim
        
        # Reshape back to (batch, d_out_1, ..., d_out_N)
        # Note: The 'batch' dimension of 'res' is effectively (batch, d_out_1, d_out_2...) rolled up in order.
        out_dims_list = [c.shape[1] for c in self.cores]
        res = res.view(*([original_bs] + out_dims_list))
        
        if not has_batch:
            res = res.squeeze(0)
            
        return res

class TTTensor(nn.Module):
    """
    Represents a Tensor Train Tensor (e.g., A tensor for observations).
    Shape: (d_1, d_2, ..., d_k)
    """
    def __init__(self, dims, ranks):
        super().__init__()
        assert len(ranks) == len(dims) + 1
        assert ranks[0] == 1 and ranks[-1] == 1
        
        self.cores = nn.ParameterList()
        self.dims = dims
        
        for i in range(len(dims)):
            # Shape: (r_in, dim_i, r_out)
            shape = (ranks[i], dims[i], ranks[i+1])
            # std = 1 / sqrt(r_in)
            core = torch.randn(shape) / (ranks[i])**0.5
            self.cores.append(nn.Parameter(core))
            
    def reconstruct(self):
        """
        Reconstructs the full dense tensor. (Expensive for large dims!)
        """
        # Contract all cores
        res = self.cores[0] # (1, d1, r1)
        for i in range(1, len(self.cores)):
            # res: (1, d1...di, ri)
            # core: (ri, d_{i+1}, r_{i+1})
            curr_core = self.cores[i]
            # Einsum: ...r, rdn -> ...dn
            # We flatten res to (..., r)
            r_in = curr_core.shape[0]
            res_flat = res.view(-1, r_in)
            curr_flat = curr_core.view(r_in, -1)
            new_res = torch.matmul(res_flat, curr_flat) # (..., d_{i+1}*r_{i+1})
            
            # Reshape back acts as concatenation of dims
            # new_shape: old_dims + [d_{i+1}, r_{i+1}]
            # We need to being careful with shapes.
            pass
            
        # Recursive implementation is cleaner?
        # Let's use the 'batch' trick from TTMatrix but starting with 1
        curr = torch.ones(1, 1, device=self.cores[0].device) # (1, r=1)
        for core in self.cores:
            # core: (r_in, d, r_out)
            # curr: (accum, r_in)
            # out: (accum, d, r_out) -> reshape (accum*d, r_out)
            r_in, d, r_out = core.shape
            # curr.view(-1, r_in) @ core.view(r_in, -1) -> (accum, d*r_out)
            curr = curr @ core.view(r_in, -1)
            curr = curr.view(-1, r_out)
            
        return curr.view(*self.dims)

    def contract_conditional(self, indices):
        """
        Efficiently gets a slice. 
        e.g., if dims = [d1, d2, d3] and indices = [None, None, idx],
        it performs the contraction fixing the last dimension.
        
        For HAI, A is 3D: (vocab, n_sem, n_syn) or similar.
        We usually want P(o | s_sem, s_syn).
        So if we fix 'o', we want the (n_sem, n_syn) matrix.
        """
        pass
        
    def forward(self, observation_idx):
        """
        Specific for the 'A' tensor in our HAI model.
        Assumes dims are [vocab_size, n_sem, n_syn].
        Returns the (n_sem, n_syn) matrix for a given observation index.
        
        This is effectively: A[o, :, :]
        """
        # A ~ Core1(o) * Core2(s_sem) * Core3(s_syn)
        # If we select o, we fix the first core.
        
        # Core1: (1, vocab, r1)
        # Select o: (1, 1, r1) -> vector (r1)
        
        c0 = self.cores[0][0, observation_idx, :] # (r1,)
        
        # Now we have a vector. We need to contract with Core2(s_sem) and Core3(s_syn)
        # to get a matrix (n_sem, n_syn).
        
        # c0 (1, r1)
        # Core2 (r1, n_sem, r2)
        # Core3 (r2, n_syn, 1)
        
        # Contraction:
        # c0 @ Core2 -> (n_sem, r2)
        # result @ Core3 -> (n_sem, n_syn)
        
        # einsum is cleanest
        t1 = torch.einsum('r, rnd -> nd', c0, self.cores[1])
        t2 = torch.einsum('nd, dsy -> nsy', t1, self.cores[2]) # Wait, dims?
        
        # Core 2: (r1, n_sem, r2)
        # Core 3: (r2, n_syn, 1)
        # t1: (n_sem, r2)
        # t2: (n_sem, n_syn, 1) -> squeeze
        
        # Let's check shapes.
        # c0: (r1)
        # Core1: (1, vocab, r1)
        # Core2: (r1, n_sem, r2)
        
        mid = torch.einsum('i, ijk -> jk', c0, self.cores[1]) # (n_sem, r2)
        final = torch.einsum('jk, kLm -> jL', mid, self.cores[2]) # (n_sem, n_syn) assuming m=1
        
        return final.squeeze(-1) # Should result in (n_sem, n_syn)
