import pandas as pd
import os

file_path = 'data/processed_injury_dataset.xls'

# Check if the file is actually a CSV file (despite .xls extension)
with open(file_path, 'rb') as f:
    header = f.read(16)
    
# If it starts with CSV-like content, treat it as CSV
if b',' in header and not header.startswith(b'\xd0\xcf\x11\xe0'):  # Not Excel magic bytes
    print("Detected CSV file with .xls extension, reading as CSV...")
    df = pd.read_csv(file_path)
else:
    print("Reading as Excel file...")
    df = pd.read_excel(file_path, engine='openpyxl')

csv_path = 'data/processed_injury_dataset.csv'
df.to_csv(csv_path, index=False)

print(f'Successfully converted to {csv_path}')
print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')


