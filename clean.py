import pandas as pd
import numpy as np
import re

df = pd.read_csv('rotten_tomatoes_critic_reviews.csv', engine='python', quotechar='"', encoding='utf-8', on_bad_lines='skip')
df = df[['review_content', 'review_type']].copy()


df.dropna(subset=['review_content'], inplace=True)
df = df[df['review_content'].str.strip() != '']
df = df[df['review_content'].str.len() > 20]

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)
    text = text.strip()
    return text

def score(text):
    if 'rotten' in text.lower():
        return 0
    elif 'fresh' in text.lower():
        return 1
    else:
        return np.nan

df['clean_review_content'] = df['review_content'].apply(clean_text)
df['label'] = df['review_type'].apply(score)

df[['clean_review_content', 'label']].to_csv('cleaned_movie_reviews.csv', index=False)

print(df.head())