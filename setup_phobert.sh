#!/bin/bash

echo "🇻🇳 Setting up PhoBERT for Vietnamese Traffic QA..."

# 1. Install dependencies
echo "📦 Installing dependencies..."
pip install transformers==4.30.0 sentencepiece protobuf

# 2. Download PhoBERT (sẽ cache local)
echo "⬇️ Downloading PhoBERT..."
python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading PhoBERT-base...')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
model = AutoModel.from_pretrained('vinai/phobert-base')
print(f'✅ Downloaded: vocab_size={len(tokenizer)}, hidden_size={model.config.hidden_size}')
"

# 3. Test tokenization
echo "🧪 Testing PhoBERT tokenization..."
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)

test_text = 'Xe ô tô có được phép rẽ trái không?'
tokens = tokenizer.tokenize(test_text)
ids = tokenizer.encode(test_text)

print(f'Input: {test_text}')
print(f'Tokens: {tokens}')
print(f'Token IDs: {ids}')
print(f'Decoded: {tokenizer.decode(ids)}')
"

echo "✅ PhoBERT setup completed!"