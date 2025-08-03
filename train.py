'''
Following the DeBERTa paper, the Train/Test split of our dataset is:

Training set: 80%

Validation set: 10%

Test set: 10%
'''

import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Loading the tokenized data
data = torch.load('deberta_tokenized_data.pt')

input_ids = data['input_ids']
attention_masks = data['attention_masks']
labels = data['labels']

# precautious attempt to ensure that labels are converted to numpy

labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

# indexing labels

indicies = np.arange(len(labels_np))

# 80% train split, 20% temporary split from entire dataset
train_idx, temp_idx = train_test_split(
    indicies, test_size=0.2, random_state=42, stratify=labels_np)

# 50% (10% from the original dataset) Validation Split and 50% (10% from the original dataset) Test Split from Temporary Split
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=labels_np[temp_idx]
)

# Creating Splits
train_inputs = input_ids[train_idx]
train_masks = attention_masks[train_idx]
train_labels = labels[train_idx]

val_inputs = input_ids[val_idx]
val_masks = attention_masks[val_idx]
val_labels = labels[val_idx]

test_inputs = input_ids[val_idx]
test_masks = attention_masks[val_idx]
test_labels = labels[val_idx]

torch.save({
    'input_ids': train_inputs,
    'attention_masks': train_masks,
    'labels': train_labels,
}, 'train_data.pt')

torch.save({
    'input_ids': val_inputs,
    'attention_masks': val_masks,
    'labels': val_labels,
}, 'validation_data.pt')

torch.save({
    'input_ids': test_inputs,
    'attention_masks': test_masks,
    'labels': test_labels,
}, 'test_data.pt')