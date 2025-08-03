import pandas as pd
import numpy as np
import torch
from transformers import DebertaTokenizer
from tqdm import tqdm

# Loading Cleaned/Labeled CSV File
df = pd.read_csv('cleaned_movie_reviews.csv')
texts = df['clean_review_content'].astype(str).tolist()
labels = df['label'].tolist()

# Initialize DeBERTa Tokenizer
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

#Batch size to avoid memory issues
batch_size = 10000
input_ids, attention_masks, all_labels = [], [], []

# Tokenization in Chuncks 

for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
    batch_texts = texts[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]

    encoding = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])
    all_labels.append(torch.tensor(batch_labels, dtype=torch.float32))

# Concatination of all the batches
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.cat(all_labels, dim=0)

# Saving all the tokenized data
torch.save({
    'input_ids': input_ids,
    'attention_masks': attention_masks,
    'labels': labels,
}, 'deberta_tokenized_data.pt')