import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hai_text.models.generative_model import DiscretePOMDP
from hai_text.models.inference import forward_backward_update
from hai_text.models.free_energy import compute_free_energy
from hai_text.models.policies import sample_next_token_via_efe
from hai_text.data.tokenizer import Tokenizer

def run_toy_task():
    """
    Task: Generate subject-verb-object sentences with constraints.
    E.g., "cat eats fish", "dog sees cat" (not "fish eats dog")
    """
    print("--- Initializing Toy POMDP ---")
    
    # Vocabulary
    vocab_list = ["cat", "dog", "fish", "eats", "sees", "chases", "."]
    tokenizer = Tokenizer(vocab_list)
    vocab_size = len(vocab_list)
    
    # State space sizes
    n_sem = 3  # Concepts: [Animal, Action, Object] or [Agent, Predicate, Patient]? 
               # Let's say: 0=Animal-Agent, 1=Action, 2=Animal-Patient
    n_syn = 3  # Syntax: 0=Subject, 1=Verb, 2=Object
    
    # --- Construct Matrices Manually for "Logic" ---
    
    # A: Likelihood P(o | s_sem, s_syn)
    # Map (sem, syn) pairs to expected tokens
    # (Animal-Agent, Subject) -> "cat", "dog"
    # (Action, Verb) -> "eats", "sees", "chases"
    # (Animal-Patient, Object) -> "fish", "cat"
    
    A = torch.zeros(vocab_size, n_sem, n_syn) + 1e-6  # Small epsilon
    
    # Sem=0 (Agent), Syn=0 (Subject) -> "cat", "dog"
    A[tokenizer.encode("cat")[0], 0, 0] = 1.0
    A[tokenizer.encode("dog")[0], 0, 0] = 1.0
    
    # Sem=1 (Action), Syn=1 (Verb) -> "eats", "sees", "chases"
    A[tokenizer.encode("eats")[0], 1, 1] = 1.0
    A[tokenizer.encode("sees")[0], 1, 1] = 1.0
    A[tokenizer.encode("chases")[0], 1, 1] = 1.0
    
    # Sem=2 (Patient), Syn=2 (Object) -> "fish", "cat"
    A[tokenizer.encode("fish")[0], 2, 2] = 1.0
    A[tokenizer.encode("cat")[0], 2, 2] = 1.0
    
    # Normalize A
    A = A / A.sum(dim=0, keepdim=True)
    
    # B: Transitions
    # We want cycle: Subject -> Verb -> Object -> Subject
    # B_syn: P(syn' | syn) should be 0->1, 1->2, 2->0
    B_syn = torch.zeros(n_syn, n_syn, vocab_size) + 1e-6
    # Deterministic syntax cycle regardless of token (simplified)
    for u in range(vocab_size):
        B_syn[1, 0, u] = 1.0 # 0->1
        B_syn[2, 1, u] = 1.0 # 1->2
        B_syn[0, 2, u] = 1.0 # 2->0
    
    # B_sem: Semantics stay roughly same? Or evolve?
    # Let's say Agent(0) -> Action(1) -> Patient(2) -> Agent(0)
    # This couples semantics to syntax strongly for this toy example
    B_sem = torch.zeros(n_sem, n_sem, vocab_size) + 1e-6
    for u in range(vocab_size):
        B_sem[1, 0, u] = 1.0
        B_sem[2, 1, u] = 1.0
        B_sem[0, 2, u] = 1.0
    
    # Normalize B
    B_sem = B_sem / B_sem.sum(dim=0, keepdim=True)
    B_syn = B_syn / B_syn.sum(dim=0, keepdim=True)
    
    # C: Preferences (Log probs) - Prefer "eats"
    C = torch.zeros(vocab_size)
    C[tokenizer.encode("eats")[0]] = 2.0  # Prefer eating
    
    # D: Initial Prior - Start at (Agent, Subject)
    D = torch.zeros(n_sem, n_syn) + 1e-6
    D[0, 0] = 1.0
    D = D / D.sum()
    
    model = DiscretePOMDP(A, B_sem, B_syn, C, D)
    
    print("--- Running Inference ---")
    
    # Test 1: Inference on observed sequence "cat eats fish"
    seq_text = "cat eats fish"
    obs = tokenizer.encode(seq_text)
    print(f"Observation: {seq_text} (IDs: {obs})")
    
    beliefs = forward_backward_update(model, obs, num_iterations=10)
    fe, complexity, accuracy = compute_free_energy(beliefs, obs, model)
    
    print(f"\nFinal Free Energy Metrics:")
    print(f"  Total FE:   {fe.item():.4f}")
    print(f"  Complexity: {complexity.item():.4f}")
    print(f"  Accuracy:   {accuracy.item():.4f}")
    
    print("\nVisualizing Beliefs (Max Sem/Syn per step):")
    for t in range(len(obs)):
        s_sem = beliefs[t][0]
        s_syn = beliefs[t][1]
        sem_idx = torch.argmax(s_sem).item()
        syn_idx = torch.argmax(s_syn).item()
        print(f"  t={t} ('{tokenizer.decode([obs[t]])[0]}'): Sem={sem_idx} ({s_sem[sem_idx]:.2f}), Syn={syn_idx} ({s_syn[syn_idx]:.2f})")
        
    print("\n--- Testing Active Generation (EFE) ---")
    
    # We want to continue from the last state of "cat eats fish"
    # Last state belief is beliefs[-1]
    
    # Let's say we are at "cat" (t=0) and want to predict next token.
    # Current belief at t=0
    current_belief = beliefs[0] # "cat" state
    
    # Candidate tokens: "eats", "sees", "fish", "dog"
    candidates_text = ["eats", "sees", "fish", "dog"]
    candidates = tokenizer.encode(" ".join(candidates_text))
    
    print(f"Current context: 'cat'")
    print(f"Candidates: {candidates_text}")
    
    # We prefer "eats" (index C set to 2.0). 
    # "sees" has C=0.
    
    next_token, socres = sample_next_token_via_efe(model, current_belief, candidates)
    next_word = tokenizer.decode([next_token])[0]
    
    print(f"Selected next token: '{next_word}'")
    
    # Check scores
    # We expect "eats" to have lower EFE (higher preference)
    for i, token_id in enumerate(candidates):
        # We need to re-eval to print or modify sample_next_token to return score
        # But sample_next_token_via_efe returns (token, scores) as I implemented it in my plan?
        # Let's check my implementation of sample_next_token_via_efe in models/policies.py
        # Yes, I implemented it to return (next_token, efe_scores).
        
        score_val = socres[i].item()
        print(f"  Token '{candidates_text[i]}': EFE = {score_val:.4f}")

    print("\n--- Validation Complete ---")

if __name__ == "__main__":
    run_toy_task()
