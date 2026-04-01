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

    unique_months = df['temp_month'].unique()
    n_months = len(unique_months)

    # 1. Initial Target per month
    target_per_month = n_total // n_months
    remainder = n_total % n_months

    sampled_indices = []

    # 2. First Pass: Take what we can from each month up to the target
    for i, month in enumerate(unique_months):
        month_indices = df[df['temp_month'] == month].index.tolist()
        month_target = target_per_month + (1 if i < remainder else 0)

        # Take either the target or the whole month if it's smaller
        n_to_take = min(len(month_indices), month_target)
        if n_to_take > 0:
            sampled_indices.extend(np.random.RandomState(42).choice(
                month_indices, n_to_take, replace=False
            ))

    # 3. Second Pass: If we are still short (because some months were tiny)
    # Fill the gap by sampling from the REMAINING pool of all other months
    if len(sampled_indices) < n_total:
        gap = n_total - len(sampled_indices)
        remaining_pool = df.drop(sampled_indices).index.tolist()

        if len(remaining_pool) >= gap:
            extra_indices = np.random.RandomState(42).choice(
                remaining_pool, gap, replace=False
            )
            sampled_indices.extend(extra_indices)

    # 4. Finalize
    final_sample = df.loc[sampled_indices].sort_values('date')
    final_sample = final_sample.drop(columns=['temp_month'])

    # Absolute Safety Check: Final Truncation
    if len(final_sample) > n_total:
        final_sample = final_sample.head(n_total)

    final_sample.to_csv(output_path, index=False)
    print(f"Successfully saved {len(final_sample)} stratified rows to {output_path}")


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
        ("ABT", 250),
        ("AMZN", 750),
        ("AVGO", 350),
        ("BEP", 100),
        ("DHR", 150),
        ("ENPH", 200),
        ("FSLR", 200),
        ("ISRG", 150),
        ("LLY", 300),
        ("META", 650),
        ("NEE", 200),
        ("NVO", 250),
        ("PLUG", 120),
        ("SNOW", 200),
        ("TSLA",800)
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