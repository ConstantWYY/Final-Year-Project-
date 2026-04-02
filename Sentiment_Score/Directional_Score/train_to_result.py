"""
Usage: python Sentiment_Score/Directional_Score/train_to_result.py --ticker ABT
"""

import os
import argparse
import pandas as pd

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
	args = parser.parse_args()

	ticker = args.ticker.upper()
	input_dir = 'Sentiment_Score/Directional_Score/Directional_Train'
	output_dir = 'Sentiment_Score/Directional_Score/Directional_Result'
	input_file = os.path.join(input_dir, f'{ticker}_train.csv')
	output_file = os.path.join(output_dir, f'{ticker}_result.csv')

	if not os.path.exists(input_file):
		raise FileNotFoundError(f'Missing input file: {input_file}')

	os.makedirs(output_dir, exist_ok=True)

	df = pd.read_csv(input_file)
	# Only keep 'date' and 'directional_sentiment' columns
	keep_cols = ['date', 'directional_sentiment']
	missing = [col for col in keep_cols if col not in df.columns]
	if missing:
		raise ValueError(f'Missing columns {missing} in {input_file}')
	result_df = df[keep_cols]

	# Save without header and index
	temp_output = output_file + '.tmp'
	result_df.to_csv(temp_output, index=False, header=False)
	os.replace(temp_output, output_file)
	print(f'Saved: {output_file}')

if __name__ == '__main__':
	main()
