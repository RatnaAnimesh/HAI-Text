class Tokenizer:
    def __init__(self, vocab_list, char_level=False):
        self.vocab = vocab_list
        # Ensure vocab is unique and sorted if list
        self.char_level = char_level
        self.token_to_id = {token: i for i, token in enumerate(vocab_list)}
        self.id_to_token = {i: token for i, token in enumerate(vocab_list)}
        
    def encode(self, text):
        if self.char_level:
            # Char level: each character is a token
            tokens = list(text)
        else:
            # Word level: simple whitespace
            tokens = text.split()
            
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]
        
    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        if self.char_level:
            return "".join(tokens)
        else:
            return " ".join(tokens)
