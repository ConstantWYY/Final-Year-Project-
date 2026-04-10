"""
Usage:
Default: python Sentiment_Score/Final_Sentiment_Score/weekly_score.py
Custom: python Sentiment_Score/Final_Sentiment_Score/weekly_score.py --tickers ABT
"""

import pandas as pd
import numpy as np
import os
import argparse

# ====================== CONFIGURATION ======================

# List of tickers to process
TICKERS = [
    'ABT', 'AMZN', 'AVGO', 'BEP', 'DHR', 'ENPH', 'FSLR', 'ISRG',
    'LLY', 'META', 'NEE', 'NVO', 'PLUG', 'SNOW', 'TSLA'
]

# File directories (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DIRECTIONAL = os.path.abspath(os.path.join(BASE_DIR, '../Directional_Score/Directional_Result'))

def process_ticker(ticker):
    csv1 = os.path.join(DIR_DIRECTIONAL, f'{ticker}_result.csv')
    output_csv = os.path.join(BASE_DIR, f'{ticker}_weekly_final_sentiment.csv')

    print(f'\nProcessing {ticker}...')
    try:
        df = pd.read_csv(csv1)
    except Exception as e:
        print(f'Error loading files for {ticker}: {e}')
        return

    # Rename columns for clarity
    df = df.rename(columns={'directional_sentiment': 'Final_Sentiment'})

    # Convert date to datetime and create week column
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').astype(str)

    # Use the absolute value of the directional score as weight, ignore zero scores
    df['Weight'] = df['Final_Sentiment'].abs()
    df_nonzero = df[df['Weight'] > 0].copy()

    # Weekly Aggregation (weighted mean, ignore zero weights)
    def weighted_mean(x):
        return np.sum(x['Final_Sentiment'] * x['Weight']) / np.sum(x['Weight']) if np.sum(x['Weight']) > 0 else 0.0

    weekly_agg = df_nonzero.groupby('week').apply(
        lambda g: pd.Series({
            'Final_Sentiment': weighted_mean(g),
            'Num_News': len(g),
            'Avg_Directional': g['Final_Sentiment'].mean(),
            'Start_Date': g['date'].min().strftime('%Y-%m-%d'),
            'End_Date': g['date'].max().strftime('%Y-%m-%d')
        })
    ).reset_index()
    weekly_agg = weekly_agg.round(4)

    # Save result
    weekly_agg.to_csv(output_csv, index=False)
    print(f' {ticker} done! File saved as {output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weekly Sentiment Score Aggregator')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to process (e.g. --tickers ABT AMZN)')
    args = parser.parse_args()

    tickers_to_process = args.tickers if args.tickers else TICKERS
    print(f'Starting sentiment aggregation (Weekly) for: {", ".join(tickers_to_process)}')
    for ticker in tickers_to_process:
        process_ticker(ticker)
    print('\nAll selected tickers processed.')
