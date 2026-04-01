"""
Usage:
Default: python DataSets\deduplicate_Transformer_Rolling.py --threshold 0.70 --lookback_days 3
Custom: python DataSets\deduplicate_Transformer_Rolling.py --ticker ABT --threshold 0.70 --lookback_days 3
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
    parser = argparse.ArgumentParser(description="Rolling N-day duplicate-news removal.")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., ABT)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Similarity threshold (0-1).")
    parser.add_argument("--lookback_days", type=int, default=3, help="How many days to look back for duplicates.")
    return parser.parse_args()


def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean_text(headline: str, summary: str) -> str:
    txt = f"{headline} {summary}".lower()
    txt = re.sub(r'^(breaking|update|live|market alert):\s*', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


def process_ticker(ticker, threshold, lookback_days):
    ticker = ticker.upper()
    input_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "DataSets", "Cleaned_Datasets", f"{ticker}_cleaned.csv"
    )
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "DataSets", "Processed_Datasets_Transformer"
    )
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.join(output_dir, f"{ticker}_processed.csv")

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    # 1. Load and Sort (Essential for rolling logic)
    df = load_and_validate(input_path)
    # Ensure 'date' column is datetime type
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    original_count = len(df)

    # 2. Vectorize All (Fast batch processing)
    print(f"Vectorizing {original_count} articles for {ticker}...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    combined_texts = [clean_text(h, s) for h, s in zip(df['headline'], df['summary'])]
    embeddings = model.encode(combined_texts, show_progress_bar=True)

    # 3. Rolling Deduplication Logic
    print(f"Applying rolling {lookback_days}-day window...")
    kept_indices = []

    for i in range(len(df)):
        current_embedding = embeddings[i].reshape(1, -1)
        current_date = df.iloc[i]['date']
        is_duplicate = False

        # Look back at previously kept unique articles
        for j in reversed(kept_indices):
            day_diff = (current_date - df.iloc[j]['date']).days
            if day_diff > lookback_days:
                break
            sim = cosine_similarity(current_embedding, embeddings[j].reshape(1, -1))[0][0]
            if sim >= threshold:
                if len(df.iloc[i]['summary']) <= len(df.iloc[j]['summary']):
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept_indices.append(i)

    # 4. Final Export
    df_dedup = df.iloc[kept_indices].copy()
    df_dedup['date'] = df_dedup['date'].dt.strftime('%Y-%m-%d')
    df_dedup.to_csv(out_name, index=False)

    print(f"\n=== Rolling Results for {ticker} ===")
    print(f"Look-back Window : {lookback_days} days")
    print(f"Threshold        : {threshold}")
    print(f"Input Rows       : {original_count}")
    print(f"Output Rows      : {len(df_dedup)}")
    print(f"Removed          : {original_count - len(df_dedup)}")
    print(f"Saved to         : {out_name}")


def main():
    args = parse_args()
    preset_ticker = [
        "ABT", "AMZN", "AVGO", "BEP", "DHR", "ENPH", "FSLR", "ISRG", "LLY", "META", "NEE", "NVO", "PLUG", "SNOW", "TSLA"
    ]
    if args.ticker:
        ticker_list = [args.ticker]
    else:
        ticker_list = preset_ticker
    for ticker in ticker_list:
        process_ticker(ticker, args.threshold, args.lookback_days)

if __name__ == "__main__":
    main()