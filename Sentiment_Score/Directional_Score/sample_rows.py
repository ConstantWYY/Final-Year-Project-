"""
Usage:
Default: python Sentiment_Score/Directional_Score/sample_rows.py
Custom: python Sentiment_Score/Directional_Score/sample_rows.py --ticker ABT --n 250
"""

import argparse
import pandas as pd
import sys
import numpy as np


def sample_stratified(input_path, output_path, n_total, random_seed=42):
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df['temp_month'] = df['date'].dt.to_period('M')

    # ✅ Handle small dataset case
    if n_total >= len(df):
        print(f"Requested {n_total}, but dataset has only {len(df)} rows.")
        df.drop(columns=['temp_month']).to_csv(output_path, index=False)
        return

    rng = np.random.RandomState(random_seed)

    unique_months = df['temp_month'].unique()
    n_months = len(unique_months)

    target_per_month = n_total // n_months
    remainder = n_total % n_months

    sampled_indices = []

    # First pass
    for i, month in enumerate(unique_months):
        month_indices = df[df['temp_month'] == month].index.tolist()
        month_target = target_per_month + (1 if i < remainder else 0)

        n_to_take = min(len(month_indices), month_target)

        if n_to_take > 0:
            sampled_indices.extend(
                rng.choice(month_indices, n_to_take, replace=False)
            )

    # Second pass (fill gap)
    gap = n_total - len(sampled_indices)

    if gap > 0:
        remaining_pool = df.drop(sampled_indices).index.tolist()

        extra = min(gap, len(remaining_pool))
        if extra > 0:
            sampled_indices.extend(
                rng.choice(remaining_pool, extra, replace=False)
            )

    final_sample = df.loc[sampled_indices].sort_values('date')
    final_sample = final_sample.drop(columns=['temp_month'])

    print(f"Final sample size: {len(final_sample)}")

    final_sample.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extract N rows using Temporal Stratification (by month) for a ticker, or use a preset array.")
    parser.add_argument("--ticker", help="Ticker symbol (e.g. ABT)")
    parser.add_argument("--n", type=int, help="Total number of rows to sample")

    args = parser.parse_args()

    import os
    # Default directory for processed datasets
    default_dir = os.path.join("DataSets", "Processed_Datasets_Transformer")
    output_dir = os.path.join(os.path.dirname(__file__), "Directional_Sample")
    os.makedirs(output_dir, exist_ok=True)

    # Hardcoded array of (ticker, n) pairs
    preset_ticker_n = [
        ("ABT", 1000),
        ("AMZN", 1000),
        ("AVGO", 1000),
        ("BEP", 1000),
        ("DHR", 1000),
        ("ENPH", 1000),
        ("FSLR", 1000),
        ("ISRG", 1000),
        ("LLY", 1000),
        ("META", 1000),
        ("NEE", 1000),
        ("NVO", 1000),
        ("PLUG", 1000),
        ("SNOW", 1000),
        ("TSLA", 1000)
    ]

    if args.ticker and args.n:
        ticker_n_list = [(args.ticker, args.n)]
    else:
        ticker_n_list = preset_ticker_n

    for ticker, n in ticker_n_list:
        input_path = os.path.join(default_dir, f"{ticker}_processed.csv")
        output_csv = os.path.join(output_dir, f"{ticker}_sample_full.csv")
        print(f"Sampling {n} rows for {ticker}...")
        sample_stratified(input_path, output_csv, n)


if __name__ == "__main__":
    main()