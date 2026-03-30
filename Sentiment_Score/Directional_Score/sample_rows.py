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
            sampled_indices.extend(np.random.RandomState(random_seed).choice(
                month_indices, n_to_take, replace=False
            ))

    # 3. Second Pass: If we are still short (because some months were tiny)
    # Fill the gap by sampling from the REMAINING pool of all other months
    if len(sampled_indices) < n_total:
        gap = n_total - len(sampled_indices)
        remaining_pool = df.drop(sampled_indices).index.tolist()
        
        if len(remaining_pool) >= gap:
            extra_indices = np.random.RandomState(random_seed).choice(
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
    parser = argparse.ArgumentParser(description="Extract N rows using Temporal Stratification (by month).")
    parser.add_argument("input_csv", help="Input CSV filename (e.g. ABT_processed.csv)")
    parser.add_argument("n", type=int, help="Total number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    import os
    # Default directory for processed datasets
    default_dir = os.path.join("DataSets", "Processed_Datasets_Transformer")
    input_path = args.input_csv
    if not os.path.isabs(input_path):
        input_path = os.path.join(default_dir, input_path)

    base = os.path.basename(input_path)
    ticker = base.split('_')[0]
    output_csv = f"{ticker}_sample_full.csv"
    sample_stratified(input_path, output_csv, args.n, args.seed)

if __name__ == "__main__":
    main()