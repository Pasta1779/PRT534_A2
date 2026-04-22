df = pd.read_parquet('cleaned_data_multiindex.parquet')
print(df.index[:3])        # Date is here
print(df.columns[:3])      # MultiIndex tuples are here
print(df.shape)            
