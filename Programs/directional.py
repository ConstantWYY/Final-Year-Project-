import pandas as pd
import re
import json
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Directional sentiment classification with config keywords.')
parser.add_argument('--input', required=True, help='Input CSV file (processed)')
parser.add_argument('--config', required=True, help='Config JSON file with keywords')
parser.add_argument('--output', default=None, help='Output CSV file (optional)')
args = parser.parse_args()

# Read the CSV
input_file = args.input
df = pd.read_csv(input_file)

# Ensure expected columns exist
expected_cols = ['date','headline','source','summary']
assert all(c in df.columns for c in expected_cols), f"Missing expected columns. Found: {df.columns.tolist()}"


# Load keywords from config file
with open(args.config, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Use *_keywords fields from config
positive_kw = config.get('positive_keywords', [])
negative_kw = config.get('negative_keywords', [])
price_move_kw = config.get('price_move_keywords', [])

# function to classify
def classify_row(headline, summary, source):
    text = ' '.join([str(headline).lower(), str(summary).lower(), str(source).lower()])
    text_norm = re.sub(r'\s+', ' ', text)

    # Option 1: Use -1, 0, 1 for sentiment
    if any(kw in text_norm for kw in positive_kw):
        return 1, 'Bullish', 'Positive keyword(s) detected.'
    if any(kw in text_norm for kw in negative_kw):
        return -1, 'Bearish', 'Negative keyword(s) detected.'
    if any(kw in text_norm for kw in price_move_kw):
        return 0, 'Neutral', 'Low-signal/price-move keyword(s) detected.'
    return 0, 'Neutral', 'No clear short-term directional signal for this article.'

def pick_evidence(headline, summary):
    # Use headline and summary as evidence, trimmed if too long
    ev = []
    def clean(q):
        if pd.isna(q):
            return None
        s = str(q).strip()
        # Limit length to 200 chars for readability
        if len(s) > 200:
            s = s[:197] + '...'
        # escape quotes
        s = s.replace('"','\\"')
        return s
    h = clean(headline)
    s = clean(summary)
    if h:
        ev.append(h)
    if s and (not h or s.lower() not in h.lower()):
        ev.append(s)
    # ensure 1-3 items
    return '[' + ', '.join(f'"{e}"' for e in ev[:3]) + ']'

# Apply classification
out_rows = []
for _, row in df.iterrows():
    score, category, rationale = classify_row(row['headline'], row['summary'], row['source'])
    evidence = pick_evidence(row['headline'], row['summary'])
    out_rows.append({
        'date': row['date'],
        'directional_score': round(score, 2),
        'category': category,
        'rationale': rationale,
        'evidence_spans': evidence
    })

out_df = pd.DataFrame(out_rows, columns=['date','directional_score','category','rationale','evidence_spans'])

# Sanity check: row count matches
assert len(out_df) == len(df)

# Save output
output_file = args.output if args.output else input_file.replace('.csv', '_directional_sentiment_output.csv')
out_df.to_csv(output_file, index=False)

# Return a small preview and path
print(output_file)
print(out_df.head(10).to_csv(index=False))
