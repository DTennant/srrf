import pandas as pd
import os, gc
import argparse
from glob import glob
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

parser = argparse.ArgumentParser()
parser.add_argument('--output', nargs='*', default=['output/baseline-larger/'])
parser.add_argument('--avg_path', type=str, default=None)

args = parser.parse_args()

# __import__("ipdb").set_trace()
df_list = []
for out in args.output:
    df_list += glob(os.path.join(out, '*.parquet'))
# Initialize a list to hold the dataframes
dfs = []

# Read the CSV files lazily and append to the list
for i, fname in enumerate(df_list):
    df = dd.read_parquet(fname)
    df['model'] = f'model_{i}'
    dfs.append(df)

# Concatenate the dataframes
combined_df = dd.concat(dfs)

# Group by 'id' and calculate the mean for each group
grouped = combined_df.groupby('id').agg({'reactivity_DMS_MaP': 'mean', 'reactivity_2A3_MaP': 'mean'})

# Compute the result (this will trigger the actual computation)
with ProgressBar():
    result = grouped.compute()
    
if args.avg_path is None:
    args.avg_path = args.output[0]

# Save to Parquet
result = result.reset_index()
result.to_parquet(os.path.join(args.avg_path, 'average_predictions.parquet'))


