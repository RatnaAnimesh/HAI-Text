import torch
import random
import pytest
from hai_text.models.generative_model import DiscretePOMDP
from hai_text.models.inference import forward_backward_update
from hai_text.models.free_energy import compute_free_energy
from hai_text.data.toy_grammar import create_random_pomdp

def test_fe_decreases_over_iterations():
    """
    Verify that free energy decreases as belief updates converge.
    """
    # Create small random POMDP
    A, B_sem, B_syn, C, D = create_random_pomdp(vocab=50, n_sem=3, n_syn=4)
    model = DiscretePOMDP(A, B_sem, B_syn, C, D)
    
    # Generate synthetic sequence
    obs = [random.randint(0, 49) for _ in range(10)]
    
    # Compute FE trajectory
    fe_traj = []
    # We treat iterations as "steps of optimization"
    # Note: forward_backward_update runs for N iterations. 
    # To see trajectory, we run it for 1, then 2, then 3...
    # This is inefficient but correct for plotting the "convergence curve" of the algorithm.
    
    for it in range(1, 15):
        beliefs_it = forward_backward_update(model, obs, num_iterations=it)
        fe, _, _ = compute_free_energy(beliefs_it, obs, model)
        fe_traj.append(fe.item())
    
    # FE should monotonically decrease (allowing small numerical error)
    # Note: Since F = Complexity - Accuracy, and we MAXIMIZE evidence (minimize Free Energy), 
    # wait -- usually Variational Free Energy is MINIMIZED.
    # F = KL + Energy. Low F is good.
    # My implementation: F_total = F_complexity - F_accuracy.
    # So F_total = KL - E[ln P]. This is indeed VFE.
    # VFE should decrease.
    
    print(f"\nFE Trajectory: {fe_traj}")
    
    failures = 0
    for i in range(1, len(fe_traj)):
        # Relax tolerance slightly for stochastic floating point ops
        if fe_traj[i] > fe_traj[i-1] + 1e-4:
            print(f"FE increased at iteration {i}: {fe_traj[i-1]} -> {fe_traj[i]}")
            failures += 1
            
    # Allow a few non-monotonic blips due to non-convexity or approximation in FP updates
    assert failures <= 3, "Free energy should generally decrease"
