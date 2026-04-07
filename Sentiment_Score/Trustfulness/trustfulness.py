"""
Usage:
Default: python Sentiment_Score/Trustfulness/trustfulness.py
Custom: python Sentiment_Score/Trustfulness/trustfulness.py --ticker ABT
"""
import pandas as pd
import os
import argparse

def process_file(input_file, output_file):
    # Define the Trustworthiness Weights (x3 Factor)
    source_weights = {
        'DowJones' : 0.90, #Institutional gold standard
        'MarketWatch': 0.90,    # Professional newsroom, high standards
        'Yahoo': 0.80,          # Large-scale aggregator
        'Finnhub': 0.4,        # Data/News API aggregator
        'SeekingAlpha': 0.3,   # Crowdsourced/User-generated (Higher bias risk)
    }
    df = pd.read_csv(input_file)
    df['x3_factor'] = df['source'].map(source_weights).fillna(0.50)
    df['x3_factor_avg'] = df['x3_factor'].mean()
    df.to_csv(output_file, index=False)
    print(f"Processed {input_file} -> {output_file}")
    print(df[['source', 'x3_factor', 'x3_factor_avg']].head())
    print(f"\nOverall Average x3_factor: {df['x3_factor'].mean():.4f}")

def main():
    parser = argparse.ArgumentParser(description="Compute trustfulness for news files.")
    parser.add_argument('--ticker', type=str, help='Ticker symbol to process (e.g. ABT)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, 'DataSets', 'Processed_Datasets_Transformer')
    output_dir = os.path.dirname(os.path.abspath(__file__))

    if args.ticker:
        ticker = args.ticker
        input_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
        output_file = os.path.join(output_dir, f"{ticker}_trustfulness.csv")
        if not os.path.exists(input_file):
            print(f"Input file for {ticker} not found: {input_file}")
            return
        process_file(input_file, output_file)
    else:
        # Batch mode: process all *_processed.csv files
        for fname in os.listdir(processed_dir):
            if fname.endswith('_processed.csv'):
                ticker = fname.split('_')[0]
                input_file = os.path.join(processed_dir, fname)
                output_file = os.path.join(output_dir, f"{ticker}_trustfulness.csv")
                process_file(input_file, output_file)
        print("Batch processing complete.")

if __name__ == "__main__":
    main()
