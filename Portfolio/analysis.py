"""
Usage: python .\Portfolio\analysis.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import os

# ==========================
# CONFIG
# ==========================
TICKERS = [
    "ABT", "AMZN", "AVGO", "DHR", "ENPH", "FSLR", "ISRG", "LLY", "META", "NEE", "NVO", "PLUG", "SNOW", "TSLA", "BEP"
]
FF3_FILE = os.path.join("Portfolio", "FF3_factor.csv")

# ==========================
# MAIN CLASS
# ==========================
class FactorSentimentStudy:


    def __init__(self, tickers):
        self.tickers = tickers
        self.factors = None
        self.stock_data = {}

    # --------------------------
    # LOAD FF3 DATA
    # --------------------------
    def load_factors(self):
        df = pd.read_csv(FF3_FILE)

        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df = df.set_index("Date")

        df = df[["Mkt-RF", "SMB", "HML", "RF"]]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        self.factors = df

    # --------------------------
    # LOAD PRICE + SENTIMENT
    # --------------------------

    def load_data(self):
        print("Downloading weekly stock data...")

        study_start = "2024-10-14" # Start Monday of the first week
        study_end = "2025-10-27"   # End Monday after the last week

        # 1. Prepare Factors
        factors = self.factors.copy()
        factors.index = pd.to_datetime(factors.index).tz_localize(None)
        # Convert index to a Weekly Period (e.g., '2024-10-14/2024-10-20')
        factors.index = factors.index.to_period('W')
        factors = factors.groupby(level=0).last()

        # 2. Download stock data
        prices_raw = yf.download(
            self.tickers,
            start=study_start,
            end=study_end,
            interval="1wk",
            progress=False
        )

        if isinstance(prices_raw.columns, pd.MultiIndex):
            prices = prices_raw["Adj Close"] if "Adj Close" in prices_raw.columns.levels[0] else prices_raw["Close"]
        else:
            prices = prices_raw["Adj Close"] if "Adj Close" in prices_raw else prices_raw["Close"]

        # Convert Price Index to Weekly Period to match Factors
        prices.index = pd.to_datetime(prices.index).tz_localize(None).to_period('W')
        returns = prices.pct_change()

        self.stock_data = {}

        for t in self.tickers:
            sent_file = os.path.join("Sentiment_Score", "Final_Sentiment_Score", f"{t}_weekly_final_sentiment.csv")
            if not os.path.exists(sent_file):
                continue

            sent_df = pd.read_csv(sent_file)
            # Use the 'week' column directly if it exists, or derive from End_Date
            sent_df["Date_Obj"] = pd.to_datetime(sent_df["End_Date"]).dt.tz_localize(None)
            
            # Create a Period Index for the week the news happened
            sent_df["News_Week_Period"] = sent_df["Date_Obj"].dt.to_period('W')
            
            # SHIFT: We want the news from this week to predict returns of the NEXT week
            sent_df["Target_Return_Week"] = sent_df["News_Week_Period"] + 1
            
            sent_df = sent_df.set_index("Target_Return_Week")

            # 3. Create Aligned Dataframe
            # Both 'returns' and 'sent_df' now use Week Periods as indexes
            df = pd.DataFrame(index=returns.index)
            df["Weekly_Return"] = returns[t]
            
            # Join using the Period Index
            df = df.join(sent_df[["Final_Sentiment"]].rename(columns={"Final_Sentiment": "Sentiment"}), how="inner")
            df = df.join(factors, how="inner")

            # Final Cleanup & Scaling
            df = df.dropna()
            df["Weekly_Return_Pct"] = df["Weekly_Return"] * 100
            df["Excess_Return"] = df["Weekly_Return_Pct"] - df["RF"]

            if len(df) >= 1:
                self.stock_data[t] = df
                print(f"[INFO] {t}: {len(df)} weeks aligned successfully.")
            else:
                print(f"[INFO] {t}: 0 weeks aligned. Sample index: {returns.index[0]} vs {sent_df.index[0]}")


    # ======================================================
    # PART 1: SENTIMENT PREDICTIVE POWER
    # ======================================================

    def sentiment_predictive(self):
        print("\n=== SENTIMENT PREDICTIVE MODEL (Weekly, Excess Return) ===")
        print("Testing: Does sentiment predict next week's EXCESS return?")

        results = []

        for t, df in self.stock_data.items():
            df = df.copy()

            # Sentiment is ALREADY aligned to the return week it predicts
            df = df.dropna(subset=["Excess_Return", "Sentiment"])

            if len(df) < 10:
                print(f"[SKIP] {t}: insufficient data")
                continue

            print(f"[Sentiment Predictive] {t}: {len(df)} weeks used")

            y = df["Excess_Return"]
            X = sm.add_constant(df["Sentiment"])

            try:
                model = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": 1}  # Newey-West for weekly data
                )

                results.append({
                    "Ticker": t,
                    "Coef": model.params.get("Sentiment", np.nan),
                    "t_stat": model.tvalues.get("Sentiment", np.nan),
                    "p_value": model.pvalues.get("Sentiment", np.nan),
                    "R2": model.rsquared
                })

            except Exception as e:
                print(f"[ERROR] Predictive model failed for {t}: {e}")

        res_df = pd.DataFrame(results)

        if not res_df.empty:
            sig = res_df[res_df["p_value"] < 0.05]
            print(f"\nSignificant stocks (p < 0.05): {len(sig)} / {len(res_df)}")
            print(res_df.sort_values("p_value"))

        return res_df


    #=======================================================
    # Part II: Does Sentiment Explain Returns AFTER Controlling for FF3?
    #========================================================
    # ======================================================
    # Step 2.1: Panel Regression
    # ======================================================
    def ff3_explanatory_panel(self):
        print("\n=== FF3 PANEL REGRESSION ===")

        # --------------------------
        # 1. STACK ALL DATA
        # --------------------------
        panel_df = []

        for t, df in self.stock_data.items():
            temp = df.copy()
            temp["Ticker"] = t
            panel_df.append(temp)

        panel_df = pd.concat(panel_df)

        # --------------------------
        # 2. CLEAN DATA
        # --------------------------
        panel_df = panel_df.dropna(subset=["Excess_Return", "Mkt-RF", "SMB", "HML"])

        print(f"Total observations: {len(panel_df)}")
        print(f"Total stocks: {panel_df['Ticker'].nunique()}")

        # --------------------------
        # 3. DEFINE MODEL
        # --------------------------
        y = panel_df["Excess_Return"]
        X = sm.add_constant(panel_df[["Mkt-RF", "SMB", "HML"]])

        # --------------------------
        # 4. RUN PANEL REGRESSION
        # --------------------------
        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 1}
        )

        print(model.summary())

        # --------------------------
        # 5. RETURN CLEAN OUTPUT
        # --------------------------
        return pd.DataFrame({
            "Coef": model.params,
            "t_stat": model.tvalues,
            "p_value": model.pvalues
        })


    # ======================================================
    # Step 2.2: COMBINED MODEL
    # ======================================================
    def combined_model(self):
        print("\n=== COMBINED MODEL (FF3 + Sentiment) ===")

        results = []

        for t, df in self.stock_data.items():
            df = df.copy()

            # Drop missing values explicitly
            df = df.dropna(subset=["Excess_Return", "Mkt-RF", "SMB", "HML", "Sentiment"])

            if len(df) < 15:
                print(f"[SKIP] {t}: insufficient data")
                continue

            y = df["Excess_Return"]

            # FF3 only model (baseline)
            X_ff3 = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
            model_ff3 = sm.OLS(y, X_ff3).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

            # FF3 + Sentiment model
            X_full = sm.add_constant(df[["Mkt-RF", "SMB", "HML", "Sentiment"]])
            model_full = sm.OLS(y, X_full).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

            results.append({
                "Ticker": t,

                # Model comparison (THIS is key)
                "R2_FF3": model_ff3.rsquared,
                "R2_Full": model_full.rsquared,
                "Delta_R2": model_full.rsquared - model_ff3.rsquared,

                # Sentiment stats
                "Sentiment_Coef": model_full.params.get("Sentiment", np.nan),
                "Sentiment_t": model_full.tvalues.get("Sentiment", np.nan),
                "Sentiment_p": model_full.pvalues.get("Sentiment", np.nan),

                # Alpha (important in asset pricing)
                "Alpha": model_full.params.get("const", np.nan),
            })

            print(f"[Combined] {t}: ΔR² = {model_full.rsquared - model_ff3.rsquared:.4f}, p = {model_full.pvalues.get('Sentiment', np.nan):.4f}")

        res_df = pd.DataFrame(results)

        if not res_df.empty:
            sig = res_df[res_df["Sentiment_p"] < 0.05]
            print(f"\nSignificant sentiment (with FF3 controls): {len(sig)} / {len(res_df)}")
            print(res_df.sort_values("Sentiment_p"))

        return res_df


    # ======================================================
    # PART III: Is Sentiment a Priced Risk Factor?
    # ======================================================
    def risk_factor_test(self):
        """
        Asset pricing test: Is sentiment a priced risk factor?
        """
        print("\n====================")
        print("PART III: RISK FACTOR TEST")
        print("====================")

        # Build df_all by stacking all stock data
        df_list = []
        for t, df in self.stock_data.items():
            temp = df.copy()
            temp = temp.rename(columns={
                "Excess_Return": "excess_ret",
                "Mkt-RF": "MKT_RF"
            })
            temp["Ticker"] = t
            temp = temp.reset_index().rename(columns={"index": "date"})
            df_list.append(temp)
        df_all = pd.concat(df_list, ignore_index=True)

        # STEP 1: Construct Sentiment Factor (High - Low)
        def construct_sentiment_factor(df, low_q=0.3, high_q=0.7):
            sent_factor = []
            for d, g in df.groupby("date"):
                if g["Sentiment"].isna().all():
                    continue
                high = g[g["Sentiment"] >= g["Sentiment"].quantile(high_q)]
                low  = g[g["Sentiment"] <= g["Sentiment"].quantile(low_q)]
                if len(high) == 0 or len(low) == 0:
                    continue
                sent_ret = high["excess_ret"].mean() - low["excess_ret"].mean()
                sent_factor.append({"date": d, "SENT": sent_ret})
            return pd.DataFrame(sent_factor).set_index("date")

        sent_factor = construct_sentiment_factor(df_all)

        # STEP 2: Merge with FF3 Factors
        ff3_factors = (
            df_all[["date", "MKT_RF", "SMB", "HML"]]
            .drop_duplicates()
            .set_index("date")
        )
        factor_df = ff3_factors.join(sent_factor, how="inner")

        # STEP 3: Time-Series Test
        Y = factor_df["SENT"]
        X = sm.add_constant(factor_df[["MKT_RF", "SMB", "HML"]])
        sent_ts_model = sm.OLS(
            Y, X
        ).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4}
        )
        print("\n--- Sentiment Factor Time-Series Regression ---")
        print(sent_ts_model.summary())

        # STEP 4: First-Pass Regressions (Estimate Betas)
        betas = []
        for ticker, g in df_all.groupby("Ticker"):
            # Drop factor columns from g to avoid overlap
            g_nofactors = g.drop(columns=[col for col in ["MKT_RF", "SMB", "HML"] if col in g.columns])
            merged = g_nofactors.set_index("date").join(factor_df, how="inner")
            if len(merged) < 30:
                continue
            Y = merged["excess_ret"]
            X = sm.add_constant(
                merged[["MKT_RF", "SMB", "HML", "SENT"]]
            )
            res = sm.OLS(Y, X).fit()
            betas.append({
                "Ticker": ticker,
                "Mean_Return": Y.mean(),
                "Beta_MKT": res.params["MKT_RF"],
                "Beta_SMB": res.params["SMB"],
                "Beta_HML": res.params["HML"],
                "Beta_SENT": res.params["SENT"]
            })
        beta_df = pd.DataFrame(betas)

        # STEP 5: Cross-Sectional Pricing Test
        Y = beta_df["Mean_Return"]
        X = sm.add_constant(
            beta_df[["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_SENT"]]
        )
        pricing_model = sm.OLS(Y, X).fit()
        print("\n--- Cross-Sectional Pricing Test (Fama–MacBeth style) ---")
        print(pricing_model.summary())

# ==========================
# RUN EVERYTHING
# ==========================
if __name__ == "__main__":

    study = FactorSentimentStudy(TICKERS)

    study.load_factors()
    study.load_data()


    sent_results = study.sentiment_predictive()
    ff3_results = study.ff3_explanatory_panel()
    combined_results = study.combined_model()
    risk_factor_result = study.risk_factor_test()