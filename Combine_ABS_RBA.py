"""
combine_tables_with_rba.py
--------------------------
Extends the existing combine_tables.py to also left-join the RBA cash rate
quarterly data onto the combined SLCI multi-index parquet.

Run ORDER:
  1. etl_table2.py          → cleaned_data_table2.parquet
  2. etl_table3.py          → cleaned_data_table3.parquet
  3. etl_rba_cashrate.py    → cleaned_data_rba.parquet          (NEW)
  4. combine_tables_with_rba.py → cleaned_data_multiindex.parquet (replaces combine_tables.py)

Join logic:
  - The SLCI data uses period-end dates (e.g. 2025-12-31).
  - The RBA quarterly data is also resampled to period-end (QE).
  - We align on exact Date match. Any SLCI quarter with no RBA row gets NaN
    for cash rate columns (this should not happen if both cover the same range).
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

TABLE2_FILE = 'cleaned_data_table2.parquet'
TABLE3_FILE = 'cleaned_data_table3.parquet'
RBA_FILE    = 'cleaned_data_rba.parquet'
OUTPUT_FILE = 'cleaned_data_multiindex.parquet'


def build_multiindex(parquet_path: str) -> pd.DataFrame:
    """
    Load a flat parquet and convert columns to a 3-level MultiIndex.
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


#  1. Load and convert SLCI tables 
print("Loading Table 2 (ABS SLCI commodity groups)...")
df_t2 = build_multiindex(TABLE2_FILE)
print(f"  Shape: {df_t2.shape}")

print("Loading Table 3 (ABS SLCI insurance/mortgage/credit)...")
df_t3 = build_multiindex(TABLE3_FILE)
print(f"  Shape: {df_t3.shape}")

#  2. Handle overlaps between T2 and T3 
overlap = df_t2.columns.intersection(df_t3.columns)
if len(overlap) > 0:
    print(f"\nWARNING: {len(overlap)} overlapping columns — Table 2 values kept:")
    for col in overlap:
        print(f"  {col}")
    df_t3 = df_t3.drop(columns=overlap)

#  3. Combine SLCI tables 
print("\nCombining SLCI tables...")
combined = pd.concat([df_t2, df_t3], axis=1)

dupes = combined.columns.duplicated()
if dupes.any():
    print(f"WARNING: {dupes.sum()} duplicate columns — dropping")
    combined = combined.loc[:, ~dupes]

print(f"Combined SLCI shape: {combined.shape}")

#  4. Load RBA quarterly data 
print("\nLoading RBA cash rate data...")
df_rba = pd.read_parquet(RBA_FILE).set_index('Date')
print(f"  Shape: {df_rba.shape}")
print(f"  Date range: {df_rba.index.min().date()} → {df_rba.index.max().date()}")

# Wrap RBA columns in a MultiIndex so the parquet stays consistent:
#   Measure='RBA', Household='All', Commodity=<column_name>
rba_tuples = [('RBA', 'All Households', col) for col in df_rba.columns]
df_rba.columns = pd.MultiIndex.from_tuples(
    rba_tuples, names=['Measure', 'Household', 'Commodity']
)

# . Left-join RBA onto SLCI (keep all SLCI rows) 
# The SLCI index may use different day-of-month than the RBA QE index.
# Normalise both to quarter-end before joining.
combined.index = combined.index.to_period('Q').to_timestamp('Q')
df_rba.index   = df_rba.index.to_period('Q').to_timestamp('Q')

combined = combined.join(df_rba, how='left')

rba_nulls = combined[('RBA', 'All Households', 'RBA_Cash_Rate_Pct')].isna().sum()
if rba_nulls > 0:
    print(f"\nWARNING: {rba_nulls} SLCI quarters have no matching RBA row "
          f"(outside RBA data range — expected for pre-RBA-data quarters).")

#  6. Summary 
print(f"\nFinal combined shape: {combined.shape}")
print(f"Date range: {combined.index.min().date()} → {combined.index.max().date()}")

print("\nTop-level Measures in combined output:")
for m in combined.columns.get_level_values('Measure').unique():
    n = combined.xs(m, level='Measure', axis=1).shape[1]
    print(f"  {m:45s} ({n} columns)")

#  7. Save 
combined.to_parquet(OUTPUT_FILE, engine='pyarrow')
print(f"\nSaved → {OUTPUT_FILE}")
print("Done.")

#  8. Quick usage example for parwuet explorer or python
print("\n--- Example: query RBA cash rate for Employee household quarters ---")
df = pd.read_parquet(OUTPUT_FILE)
# Access RBA columns
rba_cols = [c for c in df.columns if c[0] == 'RBA']
print(df[rba_cols].tail(6).to_string())