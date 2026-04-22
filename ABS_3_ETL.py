"""
etl_table3.py
-------------
ETL for ABS SLCI Table 3 — Gross insurance, mortgage interest charges
and consumer credit, index numbers and percentage changes, by household type.

Output: cleaned_data_table3.parquet  (flat, Date column, no MultiIndex)
"""

import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE  = '/home/tim/Documents/Studies/CDU/PRT564/Assignments/Assignment2/Data3.xlsx'
OUTPUT_FILE = 'cleaned_data_table3.parquet'
META_FILE   = 'metadata_table3.json'

#  1. Load raw sheet 
df_raw = pd.read_excel(INPUT_FILE, sheet_name='Sheet1', header=None)
print(f"Raw shape: {df_raw.shape}")

#  2. Extract metadata (rows 1–9) 
metadata_rows   = df_raw.iloc[1:10, :].copy()
metadata_labels = metadata_rows.iloc[:, 0].tolist()
metadata_values = metadata_rows.iloc[:, 1:].values

metadata_dict = {}
for i, label in enumerate(metadata_labels):
    if pd.notna(label):
        values = metadata_values[i, :]
        metadata_dict[str(label).strip()] = [v for v in values if pd.notna(v)]

with open(META_FILE, 'w') as f:
    json.dump(metadata_dict, f, indent=2, default=str)
print(f"Metadata saved → {META_FILE}")

#  3. Parse column headers 
raw_headers = df_raw.iloc[0, 1:].fillna('').astype(str).tolist()

column_names = ['Date']
for h in raw_headers:
    clean = ' ; '.join(
        p.strip() for p in h.split(';') if p.strip()
    )
    column_names.append(clean if clean else f'Column_{len(column_names)}')

print(f"Columns parsed: {len(column_names)}")

#  4. Locate data start 
data_start_idx = 10
for idx in range(10, len(df_raw)):
    cell = df_raw.iloc[idx, 0]
    if isinstance(cell, str) and '1998' in cell:
        data_start_idx = idx
        break
    elif hasattr(cell, 'year') and cell.year >= 1998:
        data_start_idx = idx
        break

print(f"Data starts at row index: {data_start_idx}")

#  5. Build clean DataFrame 
df_data = df_raw.iloc[data_start_idx:, :].copy()
df_data.columns = column_names

df_data['Date'] = pd.to_datetime(df_data['Date'], errors='coerce')
df_clean = df_data.dropna(subset=['Date']).reset_index(drop=True)

for col in df_clean.columns[1:]:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = df_clean.sort_values('Date').reset_index(drop=True)

#  6. Save 
df_clean.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
print(f"Saved → {OUTPUT_FILE}")
print(f"Shape: {df_clean.shape}")
print(f"Date range: {df_clean['Date'].min().date()} → {df_clean['Date'].max().date()}")

# ── 7. Summary of commodities captured ───────────────────────────────────────
commodities = set()
for col in df_clean.columns[1:]:
    parts = [p.strip() for p in col.split(';') if p.strip()]
    if len(parts) >= 3:
        commodities.add(parts[2])
print(f"Commodities in Table 3: {sorted(commodities)}")