import argparse
import pandas as pd
import sys


def sample_csv(input_path, output_path, n_rows, random_seed=None):
    try:
        df = pd.read_csv(input_path)
        if n_rows > len(df):
            print(f"Requested {n_rows} rows, but the dataset only has {len(df)} rows. Sampling all rows.")
            n_rows = len(df)
        sample = df.sample(n=n_rows, random_state=random_seed)
        sample.to_csv(output_path, index=False)
        print(f"Sampled {n_rows} rows saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract N random rows from a CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_csv", help="Path to save the sampled CSV file")
    parser.add_argument("n", type=int, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    args = parser.parse_args()

    sample_csv(args.input_csv, args.output_csv, args.n, args.seed)


if __name__ == "__main__":
    main()
