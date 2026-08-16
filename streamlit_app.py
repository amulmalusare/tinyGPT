import os
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_blocks import Block

st.set_page_config(page_title='TinyGPT Chat', layout='wide')


@st.cache_data
def build_corpus_and_vocab():
    corpus = [
        'hello friends how are you',
        'the tea is very hot',
        'my name is Aarohi',
        'the roads of Delhi are busy',
        'it is raining in Mumbai',
        'the train is late again',
        'i love eating samosas and drinking tea',
        'holi is my favorite festival',
        'diwali brings lights and sweets',
        'india won the cricket match',
        'the sun rises in the east and sets in the west',
        'the moon orbits around the earth',
    ]
    corpus = [sentence + ' <END>' for sentence in corpus]
    text = ' '.join(corpus)
    words = sorted(set(text.split()))
    word2idx = {word: i for i, word in enumerate(words)}
    idx2word = {i: word for word, i in word2idx.items()}
    data = torch.tensor([word2idx[word] for word in text.split()], dtype=torch.long)
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
                loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
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


def get_local_reply(prompt: str, max_new_tokens: int = 60):
    _, _, words, word2idx, idx2word, _ = build_corpus_and_vocab()

    if 'model' not in st.session_state:
        st.session_state.model = create_model(len(words), 6)

    model = st.session_state.model
    prompt_tokens = [token for token in prompt.lower().split() if token in word2idx]
    if not prompt_tokens:
        prompt_tokens = ['hello']

    input_ids = torch.tensor([[word2idx[token] for token in prompt_tokens]], dtype=torch.long)
    output = model.generate(input_ids, max_new_tokens)
    generated = ' '.join(idx2word[int(i)] for i in output[0])
    return generated


def get_openai_reply(prompt: str, model_name: str, api_key: str):
    if not api_key:
        raise ValueError('Please add your OpenAI API key in the sidebar.')

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def get_hf_reply(prompt: str, model_name: str, api_key: str):
    if not api_key:
        raise ValueError('Please add your Hugging Face API key in the sidebar.')

    headers = {'Authorization': f'Bearer {api_key}'}
    payload = {
        'inputs': prompt,
        'parameters': {'max_new_tokens': 200, 'temperature': 0.8, 'return_full_text': False},
    }
    url = f'https://api-inference.huggingface.co/models/{model_name}'
    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f'HF API error: {response.text}')

    data = response.json()
    if isinstance(data, list):
        return data[0].get('generated_text', '').strip()
    if isinstance(data, dict):
        return data.get('generated_text', '').strip() or data.get('error', '')
    return str(data)


st.title('TinyGPT Chat')
st.caption('A chat-style LLM demo with local generation and optional API backends.')

corpus, _, words, _, _, _ = build_corpus_and_vocab()

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'Hi! I am TinyGPT. Ask me anything.'}]

with st.sidebar:
    st.header('Settings')
    backend = st.selectbox('AI backend', ['Local TinyGPT', 'OpenAI', 'Hugging Face'])
    max_new_tokens = st.slider('Response length', 10, 200, 60)

    openai_key = st.text_input('OpenAI API key', type='password', value=os.getenv('OPENAI_API_KEY', ''))
    openai_model = st.text_input('OpenAI model', value='gpt-4o-mini')

    hf_key = st.text_input('Hugging Face API key', type='password', value=os.getenv('HF_API_KEY', ''))
    hf_model = st.text_input('HF model', value='google/flan-t5-small')

    if st.button('Clear chat'):
        st.session_state.messages = [{'role': 'assistant', 'content': 'Hi! I am TinyGPT. Ask me anything.'}]

    st.markdown('---')
    with st.expander('Corpus preview'):
        st.write('\n'.join(corpus))
    st.metric('Vocabulary', len(words))
    st.metric('Context size', 6)

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'].replace('\n', '<br>'), unsafe_allow_html=True)

with st.form('chat_form', clear_on_submit=True):
    user_prompt = st.text_area('Message', placeholder='Type your message here...', height=120)
    submitted = st.form_submit_button('Send')

if submitted and user_prompt.strip():
    st.session_state.messages.append({'role': 'user', 'content': user_prompt.strip()})
    with st.chat_message('user'):
        st.markdown(user_prompt.strip().replace('\n', '<br>'), unsafe_allow_html=True)

    try:
        with st.spinner('Thinking...'):
            if backend == 'Local TinyGPT':
                reply = get_local_reply(user_prompt.strip(), max_new_tokens=max_new_tokens)
            elif backend == 'OpenAI':
                reply = get_openai_reply(user_prompt.strip(), openai_model, openai_key)
            else:
                reply = get_hf_reply(user_prompt.strip(), hf_model, hf_key)
    except Exception as exc:
        reply = f'Error: {exc}'

    st.session_state.messages.append({'role': 'assistant', 'content': reply})
    with st.chat_message('assistant'):
        st.markdown(reply.replace('\n', '<br>'), unsafe_allow_html=True)

st.caption('Local mode uses the TinyGPT toy model; API mode requires valid credentials.')
