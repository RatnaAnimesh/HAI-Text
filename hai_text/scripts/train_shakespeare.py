import os
import torch
import requests
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hai_text.models.generative_model import DiscretePOMDP
from hai_text.training.dataset import TextDataset
from hai_text.training.trainer import HAITrainer
from hai_text.data.tokenizer import Tokenizer
from hai_text.models.policies import sample_next_token_via_efe

def download_tiny_shakespeare(file_path):
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        r = requests.get(url)
        with open(file_path, 'w') as f:
            f.write(r.text)
        print("Download complete.")
    else:
        print("Dataset found.")

def main():
    # 1. Prepare Data
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_dir, exist_ok=True)
    input_file = os.path.join(data_dir, 'input.txt')
    download_tiny_shakespeare(input_file)
    
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Use full text
    # text = text[:10000] 
    
    # Simple character-level tokenizer
    vocab_list = sorted(list(set(text)))
    tokenizer = Tokenizer(vocab_list, char_level=True)
    print(f"Vocab size: {len(vocab_list)}")
    
    dataset = TextDataset(text, tokenizer, seq_len=16, stride=16)
    print(f"Number of training sequences: {len(dataset)}")
    
    # 2. Initialize Model
    # Scaled up to 512x512 states (approx 260k latent states)
    # This provides capacity for complex grammar and prevents "jargon"
    model = DiscretePOMDP(
        vocab_size=len(vocab_list),
        n_sem=512,   # Latent semantic factors
        n_syn=512    # Latent syntactic factors
    )
    
    save_path = os.path.join(data_dir, 'model_v2_512.pt')
    
    # Check if model exists
    if os.path.exists(save_path):
        print(f"Loading pre-trained model from {save_path}...")
        model.load_state_dict(torch.load(save_path))
    else:
        # 3. Train
        print("Starting Training...")
        trainer = HAITrainer(model, dataset, learning_rate=0.05, batch_size=32)
        
        for epoch in range(2):  # Short training
            loss = trainer.train_epoch(epoch)
            print(f"Epoch {epoch} Complete. Avg VFE: {loss:.4f}")
            
            # Save Checkpoint
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    # 5. Generate Text (Evaluation)
    print("\n--- Generating Text ---")
    
    # Start with a prime (random seed state)
    current_belief = model.initial_belief()
    
    generated_ids = []
    # Generate 200 tokens
    for _ in range(200):
        # Generative Sampling (Standard HMM/POMDP generation)
        # 1. Sample next state from current belief and transition B
        # s_t ~ Discrete(current_belief)
        # s_t+1 ~ P(s'|s_t)
        
        # Approximate: Propagate belief distributions directly
        # P(s_next) = B @ s_curr
        # We assume "action" is just the previous token? 
        # In passive generation, there is no "control". 
        # But our model expects an `action_token` for B(u).
        # We feed the *last generated token* as the action (autoregressive).
        
        last_token = generated_ids[-1] if generated_ids else 0 # Default prime
        
        s_sem_next, s_syn_next = model.transition_prior(
            current_belief[0], current_belief[1], last_token
        )
        
        # 2. Sample observation from likelihood A
        # P(o) = sum_{s_sem, s_syn} P(o|s_sem, s_syn) P(s_sem) P(s_syn)
        # This is (A [vocab, :, :]) contracted with s_sem, s_syn
        
        # A is (vocab, n_sem, n_syn)
        # We want P(o) shape (vocab,)
        # einsum 'vxy,x,y->v'
        p_o = torch.einsum('vxy,x,y->v', model.A, s_sem_next, s_syn_next)
        
        # Sample token
        token = torch.multinomial(p_o, 1).item()
        generated_ids.append(token)
        
        # Update belief state (Propagate filter)
        # In true generation, we know the state is collapsing to the sample 
        # but maintaining the full distribution allows "beam-like" diversity?
        # Standard: propagate the distribution.
        current_belief = (s_sem_next, s_syn_next)
        
    gen_text = "".join(tokenizer.decode(generated_ids))
    print(f"Generated (Generative Sampling): {gen_text}")

if __name__ == "__main__":
    main()
