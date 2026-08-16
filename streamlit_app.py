import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_blocks import Block

st.set_page_config(page_title="TinyGPT — Streamlit UI")

@st.cache_data
def build_corpus_and_vocab():
    corpus = [
        "hello friends how are you",
        "the tea is very hot",
        "my name is Aarohi",
        "the roads of Delhi are busy",
        "it is raining in Mumbai",
        "the train is late again",
        "i love eating samosas and drinking tea",
        "holi is my favorite festival",
        "diwali brings lights and sweets",
        "india won the cricket match",
        "the sun rises in the east and sets in the west",
        "the moon orbits around the earth"
    ]
    corpus = [s + " <END>" for s in corpus]
    text = " ".join(corpus)
    words = list(set(text.split()))
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for w, i in word2idx.items()}
    data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
    return corpus, text, words, word2idx, idx2word, data


@st.cache_resource
def create_model(vocab_size, block_size, embedding_dim=32, n_heads=2, n_layers=2):
    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.position_embedding = nn.Embedding(block_size, embedding_dim)
            self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(embedding_dim)
            self.head = nn.Linear(embedding_dim, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding(idx)
            pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                idx = torch.cat((idx, next_idx), dim=1)
            return idx

    return TinyGPT()


st.title("TinyGPT — Streamlit UI")

corpus, text, words, word2idx, idx2word, data = build_corpus_and_vocab()

vocab_size = len(words)
block_size = 6

if 'model' not in st.session_state:
    st.session_state.model = create_model(vocab_size, block_size)
    st.session_state.optimizer = torch.optim.AdamW(st.session_state.model.parameters(), lr=1e-3)

model = st.session_state.model
optimizer = st.session_state.optimizer

def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


with st.sidebar:
    st.header("Training")
    epochs = st.number_input("Epochs", min_value=0, max_value=10000, value=100, step=50)
    batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=16)
    lr = st.number_input("Learning rate", value=1e-3, format="%.6f")
    if st.button("Train"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        progress = st.progress(0)
        status = st.empty()
        for step in range(int(epochs)):
            xb, yb = get_batch(int(batch_size))
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % max(1, int(epochs//10)) == 0:
                status.write(f"Step {step}, loss={loss.item():.4f}")
            progress.progress(int((step+1)/epochs*100))
        status.write("Training finished.")

with st.sidebar:
    st.header("Generation")
    seed_word = st.selectbox("Seed word", words, index=words.index("hello") if "hello" in words else 0)
    max_new = st.slider("Max new tokens", 1, 50, 15)
    if st.button("Generate"):
        idx = torch.tensor([[word2idx[seed_word]]], dtype=torch.long)
        out = model.generate(idx, max_new)
        generated = " ".join(idx2word[int(i)] for i in out[0])
        st.subheader("Generated text")
        st.write(generated)

st.subheader("Corpus")
st.write("\n".join(corpus))
