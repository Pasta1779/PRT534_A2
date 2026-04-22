"""
etl_rba_cashrate.py
-------------------
ETL for RBA Cash Rate Target data.

Source: Reserve Bank of Australia — Cash Rate Target history (cashrate.xlsx)
This is a genuinely heterogeneous data source: a different institution (RBA vs ABS),
different data type (monetary policy decisions vs household expenditure price indexes),
and different temporal structure (irregular meeting dates vs quarterly observations).

Pipeline:
  1. Load raw xlsx — dates stored as Excel serial integers
  2. Convert serial dates → proper datetime
  3. Forward-fill cash rate to quarterly period-end dates (matching ABS SLCI cadence)
  4. Derive a binary 'Rate_Change_Quarter' flag (1 = at least one rate change in that quarter)
  5. Save as cleaned_data_rba.parquet (flat, Date column, ready to left-join on SLCI Date index)

Output: cleaned_data_rba.parquet
"""

import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE  = 'cashrate.xlsx'   # <-- update path to match your local setup
OUTPUT_FILE = 'cleaned_data_rba.parquet'
META_FILE   = 'metadata_rba.json'

#  1. Load raw sheet 
df_raw = pd.read_excel(INPUT_FILE, sheet_name='Sheet1', header=0)
print(f"Raw shape: {df_raw.shape}")
print(f"Columns (raw): {df_raw.columns.tolist()}")
print(df_raw.head(3))

#  2. Normalise column names 
# The source xlsx uses a non-breaking space (\xa0) in the change column header.
# Strip and normalise all column names defensively.
df_raw.columns = [c.replace('\xa0', ' ').strip() for c in df_raw.columns]

# Rename columns for clarity
df_raw = df_raw.rename(columns={
    'Effective Date':   'RBA_Decision_Date',
    'Change% points':   'RBA_Rate_Change_ppt',
    'Cash rate target%':'RBA_Cash_Rate_Pct',
})
print(f"Columns (normalised): {df_raw.columns.tolist()}")

#  3. Fixthe dates 
# pandas may already parse dates correctly depending on the xlsx cell format.
# Defensively handle both: already-datetime objects AND Excel serial integers.
EXCEL_EPOCH = pd.Timestamp('1899-12-30')

def parse_rba_date(val):
    """Convert Excel serial int OR already-parsed datetime to pd.Timestamp."""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (int, float)):
        return EXCEL_EPOCH + pd.Timedelta(days=int(val))
    return pd.Timestamp(val)

df_raw['RBA_Decision_Date'] = df_raw['RBA_Decision_Date'].apply(parse_rba_date)
df_raw = df_raw.dropna(subset=['RBA_Decision_Date']).sort_values('RBA_Decision_Date').reset_index(drop=True)

print(f"\nDate range (raw decisions): {df_raw['RBA_Decision_Date'].min().date()} "
      f"→ {df_raw['RBA_Decision_Date'].max().date()}")
print(f"Total RBA decisions: {len(df_raw)}")

#  3. Save metadata 
metadata = {
    'source':       'Reserve Bank of Australia',
    'series':       'Cash Rate Target',
    'url':          'https://www.rba.gov.au/statistics/cash-rate/',
    'institution':  'RBA (heterogeneous — different from ABS SLCI source)',
    'date_range':   f"{df_raw['RBA_Decision_Date'].min().date()} to "
                    f"{df_raw['RBA_Decision_Date'].max().date()}",
    'n_decisions':  int(len(df_raw)),
    'note':         (
        'Dates in the source xlsx are stored as Excel serial integers. '
        'Converted using Excel epoch 1899-12-30. '
        'Forward-filled to quarterly period-end to align with ABS SLCI cadence.'
    ),
}
with open(META_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved → {META_FILE}")

#  4. Resample to quarterly (period-end, matching ABS SLCI) 
# ABS SLCI quarters end in March, June, September, December.
# Strategy:
#   a) Set RBA decision dates as the index.
#   b) Resample to Q-DEC frequency (quarters ending Dec/Mar/Jun/Sep).
#   c) For the cash rate: take the LAST value in each quarter (rate in effect
#      at quarter-end — this is what households face during that period).
#   d) For the change flag: take the SUM of absolute changes (>0 means at
#      least one move occurred in that quarter).

# Coerce to numeric — the xlsx stores these as object dtype (mixed str/float)
df_raw['RBA_Rate_Change_ppt'] = pd.to_numeric(df_raw['RBA_Rate_Change_ppt'], errors='coerce')
df_raw['RBA_Cash_Rate_Pct']   = pd.to_numeric(df_raw['RBA_Cash_Rate_Pct'],   errors='coerce')

df_rba_ts = df_raw.set_index('RBA_Decision_Date')

df_quarterly = pd.DataFrame({
    # Cash rate at end of quarter (last decision's rate, forward-filled)
    'RBA_Cash_Rate_Pct': df_rba_ts['RBA_Cash_Rate_Pct'].resample('QE').last(),
    # Net change in cash rate during the quarter (sum of individual moves)
    'RBA_Net_Change_ppt': df_rba_ts['RBA_Rate_Change_ppt'].resample('QE').sum(),
    # Count of decisions in the quarter
    'RBA_Decision_Count': df_rba_ts['RBA_Rate_Change_ppt'].resample('QE').count(),
}).reset_index()

df_quarterly = df_quarterly.rename(columns={'RBA_Decision_Date': 'Date'})

# Forward-fill quarters where RBA made no decision (rate was unchanged,
# so carry forward the last known rate — decision count stays 0)
full_q_index = pd.date_range(
    start=df_quarterly['Date'].min(),
    end=df_quarterly['Date'].max(),
    freq='QE'
)
df_quarterly = (
    df_quarterly
    .set_index('Date')
    .reindex(full_q_index)
    .assign(
        RBA_Cash_Rate_Pct  = lambda d: d['RBA_Cash_Rate_Pct'].ffill(),
        RBA_Net_Change_ppt = lambda d: d['RBA_Net_Change_ppt'].fillna(0),
        RBA_Decision_Count = lambda d: d['RBA_Decision_Count'].fillna(0).astype(int),
    )
    .reset_index()
    .rename(columns={'index': 'Date'})
)

# Binary flag: did the rate change at all this quarter?
df_quarterly['RBA_Rate_Changed'] = (df_quarterly['RBA_Net_Change_ppt'] != 0).astype(int)

# Direction flag: +1 hike, -1 cut, 0 hold
df_quarterly['RBA_Direction'] = df_quarterly['RBA_Net_Change_ppt'].apply(
    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
)

print(f"\nQuarterly shape: {df_quarterly.shape}")
print(f"Date range (quarterly): {df_quarterly['Date'].min().date()} "
      f"→ {df_quarterly['Date'].max().date()}")
print(df_quarterly.tail(8).to_string(index=False))

#  5. Save 
df_quarterly.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
print(f"\nSaved → {OUTPUT_FILE}")
print("Columns:", df_quarterly.columns.tolist())