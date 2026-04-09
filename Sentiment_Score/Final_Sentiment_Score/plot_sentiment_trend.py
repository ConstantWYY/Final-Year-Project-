"""
Usage: python Sentiment_Score/Final_Sentiment_Score/plot_sentiment_trend.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_all_sentiment_trends(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*_weekly_final_sentiment.csv'))
    n = len(csv_files)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False)
    for idx, csv_file in enumerate(csv_files):
        stock_name = os.path.basename(csv_file).split('_')[0]
        df = pd.read_csv(csv_file)
        # Try to infer the week/date and sentiment score columns
        week_col = None
        score_col = None
        for col in df.columns:
            if 'week' in col.lower() or 'date' in col.lower():
                week_col = col
            if 'final_sentiment' in col.lower() or 'sentiment' in col.lower() or 'score' in col.lower():
                score_col = col
        if week_col is None or score_col is None:
            print(f'Could not find week/date or sentiment score columns in {csv_file}.')
            print('Columns found:', df.columns.tolist())
            continue

        # yfinance code removed; only sentiment score will be plotted

        # Select every 4th week for clarity
        df_sampled = df.iloc[::4].reset_index(drop=True)
        ax = axes[idx // ncols][idx % ncols]
        # Plot only the sentiment score for each stock
        ax.plot(df_sampled[week_col], df_sampled[score_col], marker='o', label='Sentiment Score')
        ax.set_title(stock_name)
        ax.set_xlabel('Week')
        ax.set_ylabel('Sentiment Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')
    # Hide any unused subplots
    for j in range(idx + 1, nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])
    plt.tight_layout()
    plt.suptitle('Weekly Final Sentiment Score Trend for Each Stock', fontsize=16, y=1.02)
    plt.show()

if __name__ == '__main__':
    folder = os.path.dirname(os.path.abspath(__file__))
    plot_all_sentiment_trends(folder)
