"""
Rolling 3-Day Semantic Deduplication (Transformer-based)
Fixes the "Sunday-Monday" boundary gap and prevents "Mass Deletion" chaining.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EXPECTED_COLS = ['date', 'headline', 'source', 'summary']

def parse_args():
    parser = argparse.ArgumentParser(description="Rolling 3-day duplicate-news removal.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to input CSV.")
    parser.add_argument("--threshold", type=float, default=0.70, help="Similarity threshold (0-1).")
    parser.add_argument("--lookback_days", type=int, default=3, help="How many days to look back for duplicates.")
    return parser.parse_args()

def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[EXPECTED_COLS]
    df['date'] = pd.to_datetime(df['date'])
    for c in ['headline', 'source', 'summary']:
        df[c] = df[c].fillna("").astype(str)
    return df

def clean_text(headline: str, summary: str) -> str:
    # Light cleaning to remove common financial boilerplate
    txt = f"{headline} {summary}".lower()
    txt = re.sub(r'^(breaking|update|live|market alert):\s*', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def main():
    args = parse_args()
    input_path = args.csv_path
    ticker = os.path.basename(input_path).split('_')[0]
    
    # 1. Load and Sort (Essential for rolling logic)
    df = load_and_validate(input_path)
    df = df.sort_values('date').reset_index(drop=True)
    original_count = len(df)

    # 2. Vectorize All (Fast batch processing)
    print(f"Vectorizing {original_count} articles for {ticker}...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    combined_texts = [clean_text(h, s) for h, s in zip(df['headline'], df['summary'])]
    embeddings = model.encode(combined_texts, show_progress_bar=True)

    # 3. Rolling Deduplication Logic
    print(f"Applying rolling {args.lookback_days}-day window...")
    kept_indices = []
    
    for i in range(len(df)):
        current_embedding = embeddings[i].reshape(1, -1)
        current_date = df.iloc[i]['date']
        is_duplicate = False
        
        # Look back at previously kept unique articles
        # We search backwards from the most recent kept article
        for j in reversed(kept_indices):
            # Check the time gap
            day_diff = (current_date - df.iloc[j]['date']).days
            
            # If we've gone back further than the lookback window, stop checking
            if day_diff > args.lookback_days:
                break
                
            # Semantic similarity check
            sim = cosine_similarity(current_embedding, embeddings[j].reshape(1, -1))[0][0]
            if sim >= args.threshold:
                # Prioritize the one with the longer summary
                if len(df.iloc[i]['summary']) <= len(df.iloc[j]['summary']):
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            kept_indices.append(i)

    # 4. Final Export
    df_dedup = df.iloc[kept_indices].copy()
    df_dedup['date'] = df_dedup['date'].dt.strftime('%Y-%m-%d')
    
    out_name = f"{ticker}_processed.csv"
    df_dedup.to_csv(out_name, index=False)

    print(f"\n=== Rolling Results for {ticker} ===")
    print(f"Look-back Window : {args.lookback_days} days")
    print(f"Threshold        : {args.threshold}")
    print(f"Input Rows       : {original_count}")
    print(f"Output Rows      : {len(df_dedup)}")
    print(f"Removed          : {original_count - len(df_dedup)}")
    print(f"Saved to         : {out_name}")

if __name__ == "__main__":
    main()