"""
Usage:
Default: python Sentiment_Score/Directional_Score/splits.py
Custom: python Sentiment_Score/Directional_Score/splits.py --ticker ABT --chunk_size 100
"""

import argparse
import os
import pandas as pd

def split_csv(input_csv, chunk_size=100):
    # Derive ticker from filename
    base = os.path.basename(input_csv)
    ticker = base.split('_')[0]

    df = pd.read_csv(input_csv)
    # Ensure 'date', 'headline', 'summary' columns exist
    if not all(col in df.columns for col in ['date', 'headline', 'summary']):
        raise ValueError("Input CSV must contain 'date', 'headline', and 'summary' columns.")

    # Use row number as 'index' (starting from 1)
    out_df = pd.DataFrame({
        'index': range(1, len(df) + 1),
        'date': df['date'],
        'headline+summary': df['headline'].astype(str) + ' ' + df['summary'].astype(str)
    })

    total_rows = len(out_df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    output_dir = os.path.join(os.path.dirname(__file__), "Directional_Sample_Split")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk = out_df.iloc[start:end].copy()
        out_name = os.path.join(output_dir, f"{ticker}_sample_{i+1}.csv")
        chunk.to_csv(out_name, index=False, header=False)
        print(f"Saved {out_name} with {len(chunk)} rows.")

def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple files of N rows each for a ticker, or use a preset array.")
    parser.add_argument("--ticker", help="Ticker symbol (e.g. ABT)")
    parser.add_argument("--chunk_size", type=int, help="Number of rows per output file (default: 100)")
    args = parser.parse_args()

    # Default directory for input files
    default_dir = os.path.join("Sentiment_Score", "Directional_Score", "Directional_Sample")

    # Hardcoded array of (ticker, chunk_size) pairs
    preset_ticker = [
        "ABT",
        "AMZN",
        "AVGO",
        "BEP",
        "DHR",
        "ENPH",
        "FSLR",
        "ISRG",
        "LLY",
        "META",
        "NEE",
        "NVO",
        "PLUG",
        "SNOW",
        "TSLA"
    ]

    if args.ticker and args.chunk_size:
        ticker_chunk_list = [(args.ticker, args.chunk_size)]
    elif args.ticker:
        ticker_chunk_list = [(args.ticker, 100)]
    else:
        ticker_chunk_list = [(ticker, 100) for ticker in preset_ticker]

    for ticker, chunk_size in ticker_chunk_list:
        input_path = os.path.join(default_dir, f"{ticker}_sample_full.csv")
        print(f"Splitting {input_path} into chunks of {chunk_size}...")
        split_csv(input_path, chunk_size)

if __name__ == "__main__":
    main()
