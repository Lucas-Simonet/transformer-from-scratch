import os
from typing import Tuple

import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
batch_size = 32
block_size = 8
max_iter = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iter = 200

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.isfile("corpus.txt"):
    r = requests.get(url, allow_redirects=True)
    with open('corpus.txt', 'w', encoding='utf-8') as f:
        f.write(r.content.decode(encoding='utf-8'))

with open('corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))
vocab = ''.join(sorted(set(text)))
print("Vocabulary : ", vocab)
print("vocab size", len(vocab))

# Look up tables for encoding and decoding
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('My dear friends'))
print(decode(encode('My dear friends')))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Here we are define the context length of our model : we use tokens[:n-1] to predict token n


def get_batch(split: str) -> Tuple[torch.tensor, torch.tensor]:
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 4 random vector of size 8
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


xb, yb = get_batch('train')


# Implementing bigram for comparison

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size=len(vocab))
m = model.to(device)

out, loss = m(xb, yb)
print(out.shape)
print(loss)

print("Output without training : ", decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_token=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for steps in range(max_iter):
    xb,yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print("Output after training : ", decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_token=300)[0].tolist()))
