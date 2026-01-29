import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=20, stride=5):
        self.text = text
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        
        # Tokenize entire corpus
        self.tokens = tokenizer.encode(text)
        
        # Create sequences
        self.samples = []
        for i in range(0, len(self.tokens) - seq_len, stride):
            self.samples.append(self.tokens[i : i + seq_len])
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)
