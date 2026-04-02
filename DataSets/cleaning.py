"""
Usage
Default: python DataSets/cleaning.py
Custom: python DataSets/cleaning.py --ticker ABT
"""

import argparse
import os
import sys
import re
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import csv


def preprocess_csv_quote_commas(input_path: str, output_path: str):
    sep = ","
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile, delimiter=sep)
        writer = csv.writer(outfile, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
        header = next(reader)
        # Ensure header uses 'headline'
        header = ["headline" if col == "title" else col for col in header]
        writer.writerow(header)
        summary_idx = header.index("summary") if "summary" in header else 3
        headline_idx = header.index("headline") if "headline" in header else 1
        for row in reader:
            # Remove all double and single quotes from headline and summary
            if len(row) > headline_idx:
                row[headline_idx] = row[headline_idx].replace('"', '').replace("'", '')
            if len(row) > summary_idx:
                row[summary_idx] = row[summary_idx].replace('"', '').replace("'", '')
            for i in range(len(row)):
                if i != headline_idx and i != summary_idx:
                    row[i] = row[i].replace('"', '').replace("'", '')
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a financial news CSV for downstream NLP tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ticker", required=False, help="Ticker symbol (e.g. ABT)")
    return parser.parse_args()


def print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def load_and_prepare(path: str) -> pd.DataFrame:
    sep = ","
    # Load UTF-8 CSV with proper quoting to handle line breaks in fields
    try:
        df = pd.read_csv(path, sep=sep, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Could not read file {path}: {e}", file=sys.stderr)
        return pd.DataFrame()  # Return empty DataFrame on error
    # Rename 'datetime' column to 'date' if present
    if 'datetime' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'datetime': 'date'})
    return df


def standardize_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Parse with dayfirst=True; errors='coerce' -> NaT for invalid
    parsed = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    invalid_mask = parsed.isna()
    removed = df[invalid_mask].copy()

    df = df[~invalid_mask].copy()
    # Use date-only in YYYY-MM-DD format
    df["date"] = parsed[~invalid_mask].dt.strftime("%Y-%m-%d")

    return df, removed


def remove_missing_and_placeholders(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Treat blanks as missing
    df["headline"] = df["headline"].replace(r"^\s*$", pd.NA, regex=True)
    df["summary"] = df["summary"].replace(r"^\s*$", pd.NA, regex=True)

    # Drop missing headline or summary
    missing_mask = df["headline"].isna() | df["summary"].isna()
    missing_removed = df[missing_mask].copy()
    df = df[~missing_mask].copy()

    # Remove rows where headline or summary contains '#NAME?'
    placeholder_mask = (
        df["headline"].str.contains(r"#NAME\?", case=False, na=False) |
        df["summary"].str.contains(r"#NAME\?", case=False, na=False)
    )
    placeholder_removed = df[placeholder_mask].copy()
    df = df[~placeholder_mask].copy()

    removed_all = pd.concat([missing_removed, placeholder_removed], axis=0, ignore_index=True)
    return df, removed_all


def remove_zacks_promotions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_text = (
        "Looking for stock market analysis and research with proves results? "
        "Zacks.com offers in-depth financial research with over 30years of proven results."
    )
    # Build boolean mask
    s = df["summary"].fillna("").astype(str)

    full_text_mask = s.str.contains(re.escape(full_text), case=False, na=False)
    zacks_mask = s.str.contains(r"\bZacks\.com\b", case=False, na=False)
    proven_results_mask = s.str.contains(r"\bproven\s+results\b", case=False, na=False)

    remove_mask = full_text_mask | zacks_mask | proven_results_mask

    removed = df[remove_mask].copy()
    df = df[~remove_mask].copy()
    return df, removed


def drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Remove duplicates by headline
    dup_headline = df.duplicated(subset=["headline"], keep="first")
    removed_headline = df[dup_headline].copy()
    df = df[~dup_headline].copy()

    # Remove duplicates by summary
    dup_summary = df.duplicated(subset=["summary"], keep="first")
    removed_summary = df[dup_summary].copy()
    df = df[~dup_summary].copy()

    removed = pd.concat([removed_headline, removed_summary], axis=0, ignore_index=True)
    return df, removed


def assert_integrity(df: pd.DataFrame):
    # Ensure required columns and constraints hold
    required = ["date", "headline", "source", "summary"]
    assert all(c in df.columns for c in required), "Missing required columns after cleaning."
    assert df["date"].notna().all(), "Found missing dates after cleaning."
    assert df["headline"].notna().all() and df["summary"].notna().all(), \
        "Found missing headline/summary after cleaning."
    # No '#NAME?' placeholders
    assert not (
        df["headline"].fillna("").str.contains(r"#NAME\?", case=False).any() or
        df["summary"].fillna("").str.contains(r"#NAME\?", case=False).any()
    ), "Found '#NAME?' placeholders after cleaning."
    # No duplicate headline or summary
    assert not df.duplicated(subset=["headline"]).any(), \
        "Found duplicate headlines after cleaning."
    assert not df.duplicated(subset=["summary"]).any(), \
        "Found duplicate summaries after cleaning."


def main():
    args = parse_args()

    preset_ticker = [
        "ABT", "AMZN", "AVGO", "BEP", "DHR", "ENPH", "FSLR", "ISRG", "LLY", "META", "NEE", "NVO", "PLUG", "SNOW", "TSLA"
    ]

    if args.ticker:
        ticker_list = [args.ticker.upper()]
    else:
        ticker_list = preset_ticker

    for ticker in ticker_list:
        inp = os.path.join("DataSets", "Raw_Datasets", f"{ticker}_raw_dataset.csv")
        out_path = os.path.join("DataSets", "Cleaned_Datasets", f"{ticker}_cleaned.csv")

        if not os.path.isfile(inp):
            print(f"ERROR: Input file not found: {inp}", file=sys.stderr)
            continue

        # Preprocess CSV to ensure all fields with commas are quoted
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix="_preprocessed.csv") as tmpfile:
            tmp_path = tmpfile.name
        preprocess_csv_quote_commas(inp, tmp_path)

        df = load_and_prepare(tmp_path)
        total_before = len(df)

        df, _ = standardize_dates(df)
        df, _ = remove_missing_and_placeholders(df)
        df, _ = remove_zacks_promotions(df)
        df, _ = drop_duplicates(df)
        df = df.reset_index(drop=True)
        df = df[["date", "headline", "source", "summary"]]
        assert_integrity(df)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{ticker}] Total rows before cleaning: {total_before}")
        print(f"[{ticker}] Total rows after cleaning:  {len(df)}")
        print(f"[{ticker}] Exported cleaned dataset to: {out_path}")


if __name__ == "__main__":
    main()