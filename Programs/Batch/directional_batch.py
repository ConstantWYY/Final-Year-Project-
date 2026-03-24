import os
import subprocess
import glob

# Paths
config_dir = os.path.join('DataSets', 'Config_files')
processed_dir = os.path.join('DataSets', 'Processed_Datasets_TFIDF')
output_dir = os.path.join('Directional_Outputs')
os.makedirs(output_dir, exist_ok=True)

# Find all config and processed files
config_files = sorted(glob.glob(os.path.join(config_dir, '*_config.json')))
processed_files = sorted(glob.glob(os.path.join(processed_dir, '*_processed.csv')))

# Map tickers to files
config_map = {os.path.basename(f).split('_')[0]: f for f in config_files}
processed_map = {os.path.basename(f).split('_')[0]: f for f in processed_files}

# Path to directional.py
script_path = os.path.join('Programs', 'directional.py')

for ticker in config_map:
    if ticker in processed_map:
        config_file = config_map[ticker]
        processed_file = processed_map[ticker]
        output_file = os.path.join(output_dir, f'{ticker}_directional_sentiment_output.csv')
        cmd = [
            'python', script_path,
            '--input', processed_file,
            '--config', config_file,
            '--output', output_file
        ]
        print(f'Running: {" ".join(cmd)}')
        subprocess.run(cmd, check=True)
    else:
        print(f'No processed file for {ticker}, skipping.')
