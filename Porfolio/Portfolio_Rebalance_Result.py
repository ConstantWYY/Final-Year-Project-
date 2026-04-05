import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Define your portfolio tickers
TICKERS = ["ABT", "AMZN", "AVGO", "BEP", "DHR", "ENPH", "FSLR",
           "ISRG", "LLY", "META", "NEE", "NVO", "PLUG", "SNOW", "TSLA"]

BENCHMARK_TICKER = '^GSPC' # S&P 500

COMMON_FACTOR_FILE = os.path.join('Porfolio', 'FF3_factor.csv')

# JUSTIFICATION: Window=8 selected to capture fast-moving sentiment changes 
# and maximize the out-of-sample testing period given limited data history.
ROLLING_WINDOW = 15

class PortfolioStrategy:
    def __init__(self, tickers, factor_file):
        self.tickers = tickers
        self.factor_file = factor_file
        self.factors_df = None
        self.stock_data = {} 
        self.forecasts = pd.DataFrame()
        self.sp500_returns = pd.Series(dtype=float)
        self.weights_history = pd.DataFrame() 

    def load_common_factors(self):
        print(f"Loading common factors from {self.factor_file}...")
        try:
            df = pd.read_csv(self.factor_file)

            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

            # 🔴 Align FF3 dates to Friday
            df['Date'] = df['Date'] + pd.offsets.Week(weekday=4)

            df.set_index('Date', inplace=True)

            # Restrict date range
            start_date = pd.to_datetime('2024-10-14')
            end_date = pd.to_datetime('2025-10-12')
            df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

            cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            self.factors_df = df[cols].dropna()

            print("Common factors loaded successfully.")

        except Exception as e:
            print(f"Error loading factor file: {e}")

    def get_stock_data(self):
        """Downloads prices and merges with properly constructed sentiment."""
        
        start_date = self.factors_df.index.min() - pd.Timedelta(weeks=12)
        end_date = datetime.datetime.now()
        
        print(f"Downloading market data for {len(self.tickers)} stocks...")
        
        tickers_to_dl = self.tickers + [BENCHMARK_TICKER]
        
        # --- Download price data ---
        try:
            yf_data = yf.download(
                tickers_to_dl,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )['Adj Close']
        except:
            yf_data = yf.download(
                tickers_to_dl,
                start=start_date,
                end=end_date,
                progress=False
            )['Close']

        # --- Weekly returns (Friday aligned) ---
        yf_weekly = yf_data.resample('W-FRI').last()
        returns_df = yf_weekly.pct_change() * 100
        
        # Benchmark
        if BENCHMARK_TICKER in returns_df.columns:
            self.sp500_returns = returns_df[BENCHMARK_TICKER]

        # ==========================================
        # PROCESS EACH STOCK
        # ==========================================
        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            
            stock_df = self.factors_df.copy()
            
            # --- Merge returns ---
            if ticker in returns_df.columns:
                stock_df = stock_df.join(
                    returns_df[ticker].rename('Weekly_Return'),
                    how='inner'
                )
            else:
                print(f"  Warning: No price data for {ticker}")
                continue

            # ==========================================
            # LOAD & FIX SENTIMENT DATA
            # ==========================================
            senti_filename = os.path.join(
                '..', 'Sentiment_Score', 'Final_Sentiment_Score',
                f"{ticker}_weekly_final_sentiment.csv"
            )

            if os.path.exists(senti_filename):
                try:
                    senti_df = pd.read_csv(senti_filename)

                    # --- Select ONLY needed columns ---
                    senti_df = senti_df[['End_Date', 'Final_Sentiment', 'Num_News']].copy()

                    # --- Convert date ---
                    senti_df['Date'] = pd.to_datetime(senti_df['End_Date'])

                    # --- Align to Friday (VERY IMPORTANT) ---
                    senti_df['Date'] = senti_df['Date'] + pd.offsets.Week(weekday=4)

                    senti_df.set_index('Date', inplace=True)

                    # ==========================================
                    # IMPROVE SENTIMENT QUALITY
                    # ==========================================

                    # 1. Weight by number of news (confidence)
                    senti_df['Adj_Sentiment'] = (
                        senti_df['Final_Sentiment'] * np.log1p(senti_df['Num_News'])
                    )

                    # 2. Cap extreme values (reduce outliers)
                    senti_df['Adj_Sentiment'] = senti_df['Adj_Sentiment'].clip(-2, 2)

                    # 3. Normalize (z-score)
                    senti_df['Weekly_Senti'] = (
                        senti_df['Adj_Sentiment'] - senti_df['Adj_Sentiment'].mean()
                    ) / senti_df['Adj_Sentiment'].std()

                    senti_df = senti_df[['Weekly_Senti']]

                    # --- Merge with stock data ---
                    stock_df = stock_df.join(senti_df, how='left')

                except Exception as e:
                    print(f"  Error reading sentiment file: {e}")
                    stock_df['Weekly_Senti'] = 0

            else:
                print(f"  No sentiment file for {ticker}, using synthetic data")
                np.random.seed(len(ticker))
                stock_df['Weekly_Senti'] = (
                    stock_df['Mkt-RF'] * 0.1 +
                    np.random.normal(0, 0.5, len(stock_df))
                )

            # ==========================================
            # FINAL CLEANING
            # ==========================================

            # Fill missing sentiment
            stock_df['Weekly_Senti'] = stock_df['Weekly_Senti'].fillna(0)

            # 🚨 CRITICAL: Lag sentiment (avoid look-ahead bias)
            stock_df['Weekly_Senti'] = stock_df['Weekly_Senti'].shift(1)

            # Drop NA after lag
            stock_df = stock_df.dropna()

            self.stock_data[ticker] = stock_df

        print("All stock data processed successfully.")

    def construct_sentiment_factor(self, top_q=0.3, bottom_q=0.3):

        print("Constructing Sentiment Factor (Long-Short)...")

        panel_list = []

        for ticker, df in self.stock_data.items():
            temp = df[['Weekly_Return', 'Weekly_Senti']].copy()
            temp['Ticker'] = ticker
            panel_list.append(temp)

        panel_df = pd.concat(panel_list)
        panel_df = panel_df.reset_index().rename(columns={'index': 'Date'})

        factor_returns = []

        for date, group in panel_df.groupby('Date'):
            group = group.dropna()

            if len(group) < 5:
                continue

            group = group.sort_values('Weekly_Senti')

            n = len(group)

            # 🔴 FIX: avoid empty groups
            cutoff_low = max(1, int(n * bottom_q))
            cutoff_high = min(n - 1, int(n * (1 - top_q)))

            bottom = group.iloc[:cutoff_low]
            top = group.iloc[cutoff_high:]

            if len(top) == 0 or len(bottom) == 0:
                continue

            long_ret = top['Weekly_Return'].mean()
            short_ret = bottom['Weekly_Return'].mean()

            factor_returns.append({
                'Date': date,
                'SENT_FACTOR': (long_ret - short_ret)
            })

        sent_factor_df = pd.DataFrame(factor_returns).set_index('Date')

        self.factors_df = self.factors_df.join(sent_factor_df, how='inner')

        # 🔴 CRITICAL: LAG FACTOR (avoid look-ahead)
        self.factors_df['SENT_FACTOR'] = self.factors_df['SENT_FACTOR'].shift(1)

        print("Sentiment factor constructed successfully.")

    def run_rolling_model(self):

        print("Running Time-Series Forecast Models...")

        all_forecasts = {}

        for ticker, df in self.stock_data.items():

            if len(df) < ROLLING_WINDOW + 10:
                continue

            df = df.copy()

            # 🔴 Ensure full alignment
            df = df.join(
                self.factors_df[['SENT_FACTOR']],
                how='inner'
            ).dropna()

            # Excess return (still in %, consistent with FF3)
            df['Excess_Return'] = df['Weekly_Return'] - df['RF']

            exog = sm.add_constant(df[['Mkt-RF', 'SMB', 'HML', 'SENT_FACTOR']])
            endog = df['Excess_Return']

            rols = RollingOLS(endog, exog, window=ROLLING_WINDOW)
            rres = rols.fit()
            params = rres.params

            # 🔴 FIX: remove rolling mean of factors
            pred = (
                params['const'] +
                params['Mkt-RF'] * df['Mkt-RF'] +
                params['SMB'] * df['SMB'] +
                params['HML'] * df['HML'] +
                params['SENT_FACTOR'] * df['SENT_FACTOR']
            ).shift(1)

            # Smooth the signal
            pred = pred.rolling(3).mean()

            all_forecasts[ticker] = pred

        self.forecasts = pd.DataFrame(all_forecasts)

        # Remove duplicate indices before aligning
        self.factors_df = self.factors_df[~self.factors_df.index.duplicated(keep='first')]
        self.forecasts = self.forecasts[~self.forecasts.index.duplicated(keep='first')]

        # Attach RF
        self.forecasts['RF'] = self.factors_df['RF']

        # 🔴 Drop empty rows
        self.forecasts = self.forecasts.dropna(how='all')

    def rebalance_portfolio(self):
        print("Simulating Dynamic Long-Short Strategy (Signal-Weighted)...")

        valid_dates = self.forecasts.index
        initial_capital = 1_000_000
        cash = initial_capital

        history = []
        weights_records = []

        MAX_WEIGHT = 0.1
        THRESHOLD = 0.05

        for date in valid_dates:
            if date not in self.forecasts.index:
                continue

            row = self.forecasts.loc[date]
            rf_rate = row['RF'] / 100.0 if 'RF' in row else 0

            # -----------------------------
            # 1. GET RAW SIGNALS
            # -----------------------------
            signals = {t: row.get(t, 0) for t in self.tickers}

            # -----------------------------
            # 2. NORMALIZE SIGNALS (Fix #5)
            # -----------------------------
            signals_series = pd.Series(signals)

            if signals_series.std() != 0 and not signals_series.isna().all():
                signals_series = (signals_series - signals_series.mean()) / signals_series.std()

            signals = signals_series.to_dict()

            # -----------------------------
            # 3. APPLY THRESHOLD (Fix #4)
            # -----------------------------
            long_signals = {t: s for t, s in signals.items() if s > THRESHOLD}
            short_signals = {t: s for t, s in signals.items() if s < -THRESHOLD}

            sum_long = sum(long_signals.values())
            sum_short = -sum(short_signals.values())
            sum_total = sum_long + sum_short

            long_alloc = sum_long / sum_total if sum_total > 0 else 0
            short_alloc = sum_short / sum_total if sum_total > 0 else 0

            current_weights = {t: 0.0 for t in self.tickers}
            period_pnl = 0.0

            # -----------------------------
            # 4. ALLOCATE LONGS
            # -----------------------------
            if sum_long > 0:
                for t, s in long_signals.items():
                    w = long_alloc * (s / sum_long)
                    current_weights[t] = w

            # -----------------------------
            # 5. ALLOCATE SHORTS
            # -----------------------------
            if sum_short > 0:
                for t, s in short_signals.items():
                    w = -short_alloc * (abs(s) / sum_short)
                    current_weights[t] = w

            # -----------------------------
            # 6. CAP WEIGHTS (Fix #3)
            # -----------------------------
            for t in current_weights:
                current_weights[t] = np.clip(current_weights[t], -MAX_WEIGHT, MAX_WEIGHT)

            # -----------------------------
            # 7. CALCULATE RETURNS
            # -----------------------------
            for t, w in current_weights.items():
                if t in self.stock_data and date in self.stock_data[t].index:
                    r = self.stock_data[t].loc[date, 'Weekly_Return'] / 100.0
                    period_pnl += w * r

            # -----------------------------
            # 8. TRANSACTION COST (basic)
            # -----------------------------
            turnover = len(long_signals) + len(short_signals)
            cost = 0.001 * turnover
            period_pnl -= cost

            # -----------------------------
            # 9. UPDATE CAPITAL
            # -----------------------------
            if turnover > 0:
                cash = cash * (1 + period_pnl)
                current_weights['Cash'] = 0.0
            else:
                cash = cash * (1 + rf_rate)
                current_weights['Cash'] = 1.0

            history.append({'Date': date, 'Strategy': cash})
            current_weights['Date'] = date
            weights_records.append(current_weights)

        results = pd.DataFrame(history).set_index('Date')
        self.weights_history = pd.DataFrame(weights_records).set_index('Date')

        # Benchmark
        if not self.sp500_returns.empty:
            bench_ret = self.sp500_returns.reindex(results.index).fillna(0) / 100.0
            results['SP500'] = initial_capital * (1 + bench_ret).cumprod()

        return results

    def plot_analysis(self, df):
        plt.style.use('bmh')

        # Debug: Check types and values before plotting
        print('Strategy dtype:', df['Strategy'].dtype)
        print('First 5 Strategy values:', df['Strategy'].head())
        if 'SP500' in df.columns:
            print('SP500 dtype:', df['SP500'].dtype)
            print('First 5 SP500 values:', df['SP500'].head())

        # Convert to float if possible (handles object dtype with scalar values)
        df['Strategy'] = pd.to_numeric(df['Strategy'], errors='coerce')
        if 'SP500' in df.columns:
            df['SP500'] = pd.to_numeric(df['SP500'], errors='coerce')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1]})

        # --- Performance ---
        ax1.plot(df.index, df['Strategy'], label='Time Series Model + Sentiment Factor', color='blue', linewidth=2)
        if 'SP500' in df.columns:
            ax1.plot(df.index, df['SP500'], label='S&P 500', color='black', linestyle='--', alpha=0.7)
        ax1.set_title("Time Series Model + Sentiment Factor vs S&P 500")
        ax1.legend()
        ax1.grid(True)
        
        # --- Allocation (Long vs Short) ---
        w_df = self.weights_history * 100
        w_df = w_df.drop(columns=['Cash'], errors='ignore') # Focus on stock exposure
        
        # Separate Longs and Shorts for cleaner plotting
        w_pos = w_df.clip(lower=0)
        w_neg = w_df.clip(upper=0)
        
        colors = cm.tab10(np.linspace(0, 1, len(self.tickers)))
        
        # Plot Longs (Up)
        w_pos.plot(kind='bar', stacked=True, ax=ax2, color=colors, width=0.9, legend=False)
        # Plot Shorts (Down)
        w_neg.plot(kind='bar', stacked=True, ax=ax2, color=colors, width=0.9, legend=False)
        
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title("Portfolio Exposure: Long (Above 0) vs Short (Below 0)")
        ax2.set_ylabel("Allocation (%)")
        
        # X-Axis formatting
        n = len(w_df) // 10 + 1
        labels = [item.strftime('%Y-%m-%d') if i % n == 0 else '' for i, item in enumerate(w_df.index)]
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # Custom Legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=c, lw=4) for c in colors]
        ax2.legend(custom_lines, self.tickers, loc='upper left', bbox_to_anchor=(1, 1), title="Assets")
        
        plt.tight_layout()
        plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    bot = PortfolioStrategy(TICKERS, COMMON_FACTOR_FILE)

    bot.load_common_factors()
    bot.get_stock_data()

    # 🔴 NEW STEP (VERY IMPORTANT)
    bot.construct_sentiment_factor()

    bot.run_rolling_model()
    res = bot.rebalance_portfolio()
    bot.plot_analysis(res)