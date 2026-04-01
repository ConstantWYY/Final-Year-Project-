"""
Usage: python Sentiment_Score/Directional_Score/ml.py --ticker ABT
"""
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse


# For classification, expect 'directional_sentiment' column
REQUIRED_TRAIN_COLS = [
    "date",
    "headline",
    "source",
    "summary",
    "directional_sentiment",
]

REQUIRED_PREDICT_COLS = [
    "date",
    "headline",
    "source",
    "summary",
]

def validate_columns(df: pd.DataFrame, required_cols, file_path: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")

from sklearn.linear_model import LogisticRegression

def build_model(train_csv_path: str):
    train_df = pd.read_csv(train_csv_path)
    validate_columns(train_df, REQUIRED_TRAIN_COLS, train_csv_path)

    train_df["headline"] = train_df["headline"].fillna("").astype(str)
    train_df["summary"] = train_df["summary"].fillna("").astype(str)
    train_df["directional_sentiment"] = pd.to_numeric(train_df["directional_sentiment"], errors="coerce")

    train_df = train_df[train_df["directional_sentiment"].notna()].copy()
    if train_df.empty:
        raise ValueError("No valid labeled rows found in training file.")

    # Combine headline and summary into one text feature
    text_train = (train_df["headline"].astype(str) + " " + train_df["summary"].astype(str)).tolist()
    y_train = train_df["directional_sentiment"].astype(int)

    embedder = SentenceTransformer("ProsusAI/finbert")
    x_train = embedder.encode(text_train, show_progress_bar=True)

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(x_train, y_train)

    return model, embedder


def predict_for_file(
    model,
    embedder,
    input_csv_path: str,
    output_csv_path: str,
    train_csv_path: str = None,
):

    df = pd.read_csv(input_csv_path)
    validate_columns(df, REQUIRED_PREDICT_COLS, input_csv_path)

    df["headline"] = df["headline"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["date"] = df["date"].astype(str)

    # Combine headline and summary into one text feature for prediction
    text = (df["headline"].astype(str) + " " + df["summary"].astype(str)).tolist()
    X = embedder.encode(text, show_progress_bar=True)
    predicted_labels = model.predict(X)

    result_df = pd.DataFrame({
        "date": df["date"].values,
        "headline": df["headline"].values,
        "directional_sentiment": predicted_labels,
    })

    if train_csv_path and os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        train_df["headline"] = train_df["headline"].fillna("").astype(str)
        train_df["date"] = train_df["date"].astype(str)

        result_df = result_df.merge(
            train_df[["date", "headline", "directional_sentiment"]]
                .dropna(subset=["directional_sentiment"])
                .drop_duplicates(subset=["date", "headline"]),
            on=["date", "headline"],
            how="left",
            suffixes=("", "_train")
        )

        result_df["directional_sentiment"] = result_df["directional_sentiment_train"].combine_first(result_df["directional_sentiment"])
        result_df.drop(columns=["directional_sentiment_train"], inplace=True)


    final_df = result_df[["date", "directional_sentiment"]]
    if len(final_df) != len(df):
        raise AssertionError(f"Row count mismatch: input ({len(df)}) vs output ({len(final_df)})")

    temp_output_path = output_csv_path + ".tmp_check"
    final_df.to_csv(temp_output_path, index=False, header=False)
    os.replace(temp_output_path, output_csv_path)



def run_single_ticker(processed_dir, outputs_dir, train_dir, ticker):
    ticker = ticker.upper()
    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    train_file = os.path.join(train_dir, f"{ticker}_train.csv")
    output_file = os.path.join(outputs_dir, f"{ticker}_result.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed file not found: {processed_file}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    os.makedirs(outputs_dir, exist_ok=True)

    model, embedder = build_model(train_file)
    predict_for_file(
        model,
        embedder,
        processed_file,
        output_file,
        train_csv_path=train_file,
    )
    print(f"Created {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Train ML model and predict directional sentiment for a single ticker."
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol to process (e.g., ABT)",
    )

    args = parser.parse_args()

    processed_dir = "DataSets/Processed_Datasets_Transformer"
    outputs_dir = "Sentiment_Score/Directional_Score/Directional_Result"
    train_dir = "Sentiment_Score/Directional_Score/Directional_Train"

    run_single_ticker(
        processed_dir,
        outputs_dir,
        train_dir,
        args.ticker,
    )


if __name__ == "__main__":
    main()