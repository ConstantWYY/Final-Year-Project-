"""
Usage: python Sentiment_Score\Directional_Score\ml.py --ticker ABT
"""

import os
import pandas as pd
import argparse
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# =========================
# Required Columns
# =========================
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


# =========================
# Build + Train FinBERT
# =========================
def build_model(train_csv_path: str):

    df = pd.read_csv(train_csv_path)
    validate_columns(df, REQUIRED_TRAIN_COLS, train_csv_path)

    df["headline"] = df["headline"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)

    df = df[df["directional_sentiment"].notna()].copy()
    if df.empty:
        raise ValueError("No valid labeled rows found.")

    # Combine text
    df["text"] = df["headline"] + " " + df["summary"]

    # Ensure labels are integers
    df["label"] = df["directional_sentiment"].astype(int)

    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df[["text", "label"]])

    # Load FinBERT
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # Tokenization
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)

    # Train args
    training_args = TrainingArguments(
        output_dir="./finbert_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    return model, tokenizer


# =========================
# Prediction
# =========================
def predict_for_file(
    model,
    tokenizer,
    input_csv_path: str,
    output_csv_path: str,
    train_csv_path: str = None,
):

    df = pd.read_csv(input_csv_path)
    validate_columns(df, REQUIRED_PREDICT_COLS, input_csv_path)

    df["headline"] = df["headline"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["date"] = df["date"].astype(str)

    df["text"] = df["headline"] + " " + df["summary"]

    # Tokenize
    inputs = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Predict
    outputs = model(**inputs)
    preds = np.argmax(outputs.logits.detach().numpy(), axis=1)

    result_df = pd.DataFrame({
        "date": df["date"].values,
        "headline": df["headline"].values,
        "summary": df["summary"].values,
        "directional_sentiment": preds,
    })

    # =========================
    # Merge train labels (override)
    # =========================
    if train_csv_path and os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)

        train_df["headline"] = train_df["headline"].fillna("").astype(str)
        train_df["summary"] = train_df["summary"].fillna("").astype(str)
        train_df["date"] = train_df["date"].astype(str)

        result_df = result_df.merge(
            train_df[["date", "headline", "summary", "directional_sentiment"]]
                .dropna(subset=["directional_sentiment"])
                .drop_duplicates(subset=["date", "headline", "summary"]),
            on=["date", "headline", "summary"],
            how="left",
            suffixes=("", "_train")
        )

        result_df["directional_sentiment"] = result_df["directional_sentiment_train"].combine_first(
            result_df["directional_sentiment"]
        )
        result_df.drop(columns=["directional_sentiment_train"], inplace=True)

    final_df = result_df[["date", "directional_sentiment"]]

    if len(final_df) != len(df):
        raise AssertionError("Row count mismatch")

    # Safe write
    temp_output_path = output_csv_path + ".tmp"
    final_df.to_csv(temp_output_path, index=False, header=False)
    os.replace(temp_output_path, output_csv_path)


# =========================
# Run
# =========================
def run_single_ticker(processed_dir, outputs_dir, train_dir, ticker):

    ticker = ticker.upper()

    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    train_file = os.path.join(train_dir, f"{ticker}_train.csv")
    output_file = os.path.join(outputs_dir, f"{ticker}_result.csv")

    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Missing processed file: {processed_file}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Missing train file: {train_file}")

    os.makedirs(outputs_dir, exist_ok=True)

    print(f"Training FinBERT for {ticker}...")
    model, tokenizer = build_model(train_file)

    print(f"Predicting for {ticker}...")
    predict_for_file(
        model,
        tokenizer,
        processed_file,
        output_file,
        train_csv_path=train_file,
    )

    print(f"Created {output_file}")


# =========================
# Main
# =========================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)

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