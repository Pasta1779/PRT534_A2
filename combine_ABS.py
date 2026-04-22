"""
combine_tables.py
-----------------
Combines cleaned_data_table2.parquet and cleaned_data_table3.parquet
into a single cleaned_data_multiindex.parquet with a 3-level MultiIndex:
    Measure → Household → Commodity

Run AFTER etl_table2.py and etl_table3.py have both completed.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

TABLE2_FILE = '/home/tim/Documents/Studies/CDU/PRT564/Assignments/Assignment2/cleaned_data_table2.parquet'
TABLE3_FILE = '/home/tim/Documents/Studies/CDU/PRT564/Assignments/Assignment2/cleaned_data_table3.parquet'
OUTPUT_FILE = '/home/tim/Documents/Studies/CDU/PRT564/Assignments/Assignment2/cleaned_data_multiindex.parquet'


def build_multiindex(parquet_path: str) -> pd.DataFrame:
    """
    Load a flat parquet file and convert its columns to a 3-level MultiIndex.
    Column format expected:  "Measure ; Household ; Commodity"
    Date becomes the DataFrame index.
    """
    df = pd.read_parquet(parquet_path)
    df = df.set_index('Date')

    tuples = []
    for col in df.columns:
        parts = [p.strip() for p in col.split(';') if p.strip()]
        if len(parts) == 3:
            tuples.append((parts[0], parts[1], parts[2]))
        else:
            print(f"  WARNING — unexpected format in {parquet_path}: {repr(col)}")
            tuples.append((col, 'Unknown', 'Unknown'))

    df.columns = pd.MultiIndex.from_tuples(
        tuples, names=['Measure', 'Household', 'Commodity']
    )
    return df


#  1. Load and convert both tables 
print("Loading Table 2...")
df_t2 = build_multiindex(TABLE2_FILE)
print(f"  Shape: {df_t2.shape}")

print("Loading Table 3...")
df_t3 = build_multiindex(TABLE3_FILE)
print(f"  Shape: {df_t3.shape}")

#  2. Check for column overlaps before combining 
overlap = df_t2.columns.intersection(df_t3.columns)
if len(overlap) > 0:
    print(f"\nWARNING: {len(overlap)} overlapping columns found — Table 2 values kept:")
    for col in overlap:
        print(f"  {col}")
    df_t3 = df_t3.drop(columns=overlap)

#  3. Combine on Date index 
print("\nCombining tables...")
combined = pd.concat([df_t2, df_t3], axis=1)

#  4. Verify 
dupes = combined.columns.duplicated()
if dupes.any():
    print(f"WARNING: {dupes.sum()} duplicate columns after concat — dropping duplicates")
    combined = combined.loc[:, ~dupes]

print(f"\nCombined shape: {combined.shape}")
print(f"Date range: {combined.index.min().date()} → {combined.index.max().date()}")

print("\nMeasures:")
for m in combined.columns.get_level_values('Measure').unique():
    print(f"  {m}")

print("\nHousehold types:")
for h in combined.columns.get_level_values('Household').unique():
    print(f"  {h}")

print("\nCommodities (Table 2):")
t2_commodities = df_t2.columns.get_level_values('Commodity').unique()
for c in sorted(t2_commodities):
    print(f"  {c}")

print("\nCommodities (Table 3 additions):")
t3_commodities = df_t3.columns.get_level_values('Commodity').unique()
for c in sorted(t3_commodities):
    print(f"  {c}")

#  5. Save 
combined.to_parquet(OUTPUT_FILE, engine='pyarrow')
print(f"\nSaved → {OUTPUT_FILE}")
print("Done.")