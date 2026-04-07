"""
Usage:
Default: python Sentiment_Score/Relevance/relevance.py
Custom: python Sentiment_Score/Relevance/relevance.py --ticker ABT
"""

import re
import csv
import json
import sys
import os
import argparse

# =========================
# Load stock-specific config
# =========================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Relevance computation
# =========================
def compute_relevance(row: dict, config: dict) -> float:
    headline = row.get("headline", "").lower()
    summary  = row.get("summary", "").lower()
    text = f"{headline} {summary}"

    aliases = [a.lower() for a in config.get("aliases", [])]
    products = [p.lower() for p in config.get("products", [])]
    macro_keywords = [k.lower() for k in config.get("macro_keywords", [])]
    indirect_keywords = [k.lower() for k in config.get("indirect_keywords", [])]
    direct_keywords = [k.lower() for k in config.get("direct_keywords", [])]
    low_signal_phrases = [p.lower() for p in config.get("low_signal_phrases", [])]

    # -----------------------------
    # Precompute signals
    # -----------------------------
    mention_count = sum(
        len(re.findall(rf"\b{re.escape(a)}\b", text))
        for a in aliases
    )
    
    firm_in_headline = any(
        re.search(rf"\b{re.escape(a)}\b", headline)
        for a in aliases
    )

    product_mentioned = any(
        re.search(rf"\b{re.escape(p)}\b", text)
        for p in products
    )

    macro_hit = any(k in text for k in macro_keywords)
    indirect_hit = any(k in text for k in indirect_keywords)
    direct_hit = any(k in text for k in direct_keywords)
    low_signal_hit = any(p in text for p in low_signal_phrases)

    # -----------------------------
    # STEP 1: Macro override
    # -----------------------------
    if macro_hit and not firm_in_headline and mention_count == 0:
        return 0.0

    # -----------------------------
    # STEP 2: Score accumulation
    # -----------------------------
    score = 0.0

    # Strong signal: firm in headline
    if firm_in_headline:
        score += 0.6

    # Mentions in text (scaled)
    score += min(mention_count * 0.2, 0.6)

    # Direct business/financial keywords
    if direct_hit:
        score += 0.5

    # Product-level signal
    if product_mentioned:
        score += 0.4

    # Industry / indirect context
    if indirect_hit:
        score += 0.2

    # Low-signal penalty
    if low_signal_hit:
        score -= 0.4

    # -----------------------------
    # STEP 3: Clamp score
    # -----------------------------
    score = max(0.0, min(score, 1.0))

    return score


# =========================
# Main processing pipeline
# =========================

def process(input_csv: str, output_csv: str, config_path: str):
    # Always resolve config_path relative to the Config_files directory next to this script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "Config_files")
    # Extract ticker from input_csv filename
    ticker = os.path.basename(input_csv).split("_")[0]
    config_file = f"{ticker}_config.json"
    config_full_path = os.path.join(config_dir, config_file)
    config = load_config(config_full_path)

    results = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("date", "")
            relevance = compute_relevance(row, config)
            results.append((date, relevance))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "relevance"])
        for r in results:
            writer.writerow(r)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute relevance for news files.")
    parser.add_argument("--ticker", type=str, help="Ticker symbol to process (e.g. ABT)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "DataSets", "Processed_Datasets_Transformer")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(output_dir, "Config_files")

    if args.ticker:
        ticker = args.ticker
        input_csv = os.path.join(processed_dir, f"{ticker}_processed.csv")
        output_csv = os.path.join(output_dir, f"{ticker}_relevance.csv")
        if not os.path.exists(input_csv):
            print(f"Input file for {ticker} not found: {input_csv}")
            sys.exit(1)
        print(f"Processing {ticker}...")
        process(input_csv, output_csv, None)
        print(f"Done: {output_csv}")
    else:
        # Default: batch mode
        for fname in os.listdir(processed_dir):
            if fname.endswith(".csv"):
                input_csv = os.path.join(processed_dir, fname)
                ticker = fname.split("_")[0]
                output_csv = os.path.join(output_dir, f"{ticker}_relevance.csv")
                # Check config exists before processing
                config_json = os.path.join(config_dir, f"{ticker}_config.json")
                if not os.path.exists(config_json):
                    print(f"Config for {ticker} not found, skipping {fname}.")
                    continue
                print(f"Processing {fname}...")
                process(input_csv, output_csv, None)
        print("Batch processing complete.")