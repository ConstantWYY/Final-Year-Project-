import os
import subprocess
import glob


# Paths
cleaned_dir = os.path.join('DataSets', 'Cleaned_Datasets')
processed_dir = os.path.join('DataSets', 'Processed_Datasets_Transformer')
os.makedirs(processed_dir, exist_ok=True)

# Find all cleaned dataset files
cleaned_files = sorted(glob.glob(os.path.join(cleaned_dir, '*_cleaned.csv')))

# Path to deduplicate_Transformer_Rolling.py
script_path = os.path.join('Programs', 'deduplicate_Transformer_Rolling.py')

for cleaned_file in cleaned_files:
    ticker = os.path.basename(cleaned_file).split('_')[0]
    print(f'Processing {ticker}...')
    cmd = [
        'python', script_path,
        '--csv-path', cleaned_file
    ]
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
