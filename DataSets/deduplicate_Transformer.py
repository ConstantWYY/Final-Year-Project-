"""
Usage:
Default: python DataSets/deduplicate_Transformer.py
Custom: python DataSets\deduplicate_Transformer.py --ticker ABT --threshold 0.80
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weekly semantic deduplication of financial news."
    )
    parser.add_argument(
        "--ticker", type=str, help="Ticker symbol (e.g., ABT)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Cosine similarity threshold for near-duplicate detection."
    )
    return parser.parse_args()


def clean_text(headline: str, summary: str) -> str:
    text = f"{headline} {summary}".lower()
    text = re.sub(r'^(breaking|update|live|market alert):\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_ticker(ticker, threshold):
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

    output_path = os.path.join(output_dir, f"{ticker}_processed.csv")

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    # Load data
    df = pd.read_csv(input_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    original_count = len(df)

    # Add week identifier
    df["week"] = df["date"].dt.to_period("W").astype(str)

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    kept_rows = []

    print(f"Processing weekly deduplication for {ticker}...")

    # Process each week independently
    for week, week_df in df.groupby("week"):
        week_df = week_df.sort_values("date").reset_index()

        texts = [
            clean_text(h, s)
            for h, s in zip(week_df["headline"], week_df["summary"])
        ]

        embeddings = model.encode(texts)

        kept_indices = []

        for i in range(len(week_df)):
            current_emb = embeddings[i].reshape(1, -1)
            is_duplicate = False

            for j in kept_indices:
                sim = cosine_similarity(
                    current_emb,
                    embeddings[j].reshape(1, -1)
                )[0][0]

                if sim >= threshold:
                    is_duplicate = True
                    break

            # Keep earliest article only
            if not is_duplicate:
                kept_indices.append(i)

        kept_rows.extend(week_df.loc[kept_indices].to_dict("records"))


    df_dedup = pd.DataFrame(kept_rows)
    df_dedup["date"] = pd.to_datetime(df_dedup["date"]).dt.strftime("%Y-%m-%d")
    if "index" in df_dedup.columns:
        df_dedup = df_dedup.drop(columns=["index"])
    df_dedup.to_csv(output_path, index=False)

    print(f"\n=== Weekly Deduplication Results for {ticker} ===")
    print(f"Similarity Threshold : {threshold}")
    print(f"Input Rows          : {original_count}")
    print(f"Output Rows         : {len(df_dedup)}")
    print(f"Removed             : {original_count - len(df_dedup)}")
    print(f"Saved to            : {output_path}")


def main():
    args = parse_args()

    preset_tickers = [
        "ABT", "AMZN", "AVGO", "BEP", "DHR", "ENPH", "FSLR",
        "ISRG", "LLY", "META", "NEE", "NVO", "PLUG", "SNOW", "TSLA"
    ]

    tickers = [args.ticker] if args.ticker else preset_tickers

    for ticker in tickers:
        process_ticker(ticker, args.threshold)


if __name__ == "__main__":
    main()