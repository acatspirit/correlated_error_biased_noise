import os
import glob
import pandas as pd

results_dir = "results"

# Find all result files
csv_files = glob.glob(os.path.join(results_dir, "result_d*_eta*_p*.csv"))

if not csv_files:
    raise ValueError("No result files found in results/")

print(f"Found {len(csv_files)} files.")

merged_df = None  # will initialize after reading first file

for file in csv_files:
    df = pd.read_csv(file)

    if merged_df is None:
        # Initialize with correct columns
        merged_df = pd.DataFrame(columns=df.columns)

    # Since each file has one row, grab that row as dict
    row_dict = df.iloc[0].to_dict()

    # Add row using .loc
    merged_df.loc[len(merged_df)] = row_dict

# Optional: sort results nicely
merged_df = merged_df.sort_values(by=["d", "eta", "p", "cutoff"]).reset_index(drop=True)

merged_df.to_csv("cutoff_results.csv", index=False)

print("Merged file written to cutoff_results.csv")