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
ALPHA = 1.0  # Exponent for Relevance
BETA = 1.0   # Exponent for Trustfulness

# List of tickers to process
TICKERS = [
    'ABT', 'AMZN', 'AVGO', 'BEP', 'DHR', 'ENPH', 'FSLR', 'ISRG',
    'LLY', 'META', 'NEE', 'NVO', 'PLUG', 'SNOW', 'TSLA'
]

# File directories (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DIRECTIONAL = os.path.abspath(os.path.join(BASE_DIR, '../Directional_Score/Directional_Result'))
DIR_RELEVANCE = os.path.abspath(os.path.join(BASE_DIR, '../Relevance'))
DIR_TRUSTFULNESS = os.path.abspath(os.path.join(BASE_DIR, '../Trustfulness'))
OUTPUT_DIR = BASE_DIR

def process_ticker(ticker):
    csv1 = os.path.join(DIR_DIRECTIONAL, f'{ticker}_result.csv')
    csv2 = os.path.join(DIR_RELEVANCE, f'{ticker}_relevance.csv')
    csv3 = os.path.join(DIR_TRUSTFULNESS, f'{ticker}_trustfulness.csv')
    output_csv = os.path.join(OUTPUT_DIR, f'{ticker}_weekly_final_sentiment.csv')

    print(f'\nProcessing {ticker}...')
    try:
        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)
        df3 = pd.read_csv(csv3)
    except Exception as e:
        print(f'Error loading files for {ticker}: {e}')
        return

    # Rename columns for clarity
    df1 = df1.rename(columns={'directional_sentiment': 'x1_Directional'})
    df2 = df2.rename(columns={'relevance': 'x2_Relevance'})
    df3 = df3.rename(columns={'x3_factor': 'x3_Trustfulness'})

    # Merge on 'date'
    try:
        df = df1.merge(df2, on='date', how='inner')
        df = df.merge(df3, on='date', how='inner')
    except Exception as e:
        print(f'❌ Error merging data for {ticker}: {e}')
        return

    # Compute Final Sentiment
    df['Final_Sentiment'] = (
        df['x1_Directional'] *
        (df['x2_Relevance'] ** ALPHA) *
        (df['x3_Trustfulness'] ** BETA)
    )
    df['Magnitude_Weight'] = np.abs(df['x2_Relevance']) * np.abs(df['x3_Trustfulness'])

    # Convert date to datetime and create week column
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').astype(str)

    # Weekly Aggregation
    weekly_agg = df.groupby('week').apply(
        lambda g: pd.Series({
            'Final_Sentiment': np.sum(g['Final_Sentiment'] * g['Magnitude_Weight']) /
                              np.sum(g['Magnitude_Weight']) if np.sum(g['Magnitude_Weight']) > 0 else 0.0,
            'Num_News': len(g),
            'Avg_Relevance': g['x2_Relevance'].mean(),
            'Avg_Trustfulness': g['x3_Trustfulness'].mean(),
            'Avg_Directional': g['x1_Directional'].mean(),
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
