import os
import glob
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True) # should be cluster_ler_chunks
    parser.add_argument("--outfile", type=str, required=True) # should be ler_below_threshold.csv
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.indir}")

    # Read new chunk files
    dfs = [pd.read_csv(f) for f in files]
    new_df = pd.concat(dfs, ignore_index=True)

    group_cols = ["l", "d", "CD_type", "mem_type", "eta", "p"]

    # Aggregate NEW data
    new_agg = (
        new_df.groupby(group_cols, as_index=False)
        .agg(
            total_shots=("shots_in_chunk", "sum"),
            total_logical_errors_pycorr=("num_logical_errors_pycorr", "sum"),
            total_logical_errors_my_corr=("num_logical_errors_my_corr", "sum"),
            num_chunks_found=("chunk_index", "count"),
            total_elapsed_seconds=("elapsed_seconds", "sum"),
        )
    )

    # If outfile exists, merge with old data
    if os.path.exists(args.outfile):
        print(f"Appending to existing file: {args.outfile}")
        old_agg = pd.read_csv(args.outfile)

        # Combine old + new
        combined = pd.concat([old_agg, new_agg], ignore_index=True)

        # Re-aggregate to correctly merge overlapping rows
        agg_df = (
            combined.groupby(group_cols, as_index=False)
            .agg(
                total_shots=("total_shots", "sum"),
                total_logical_errors_pycorr=("total_logical_errors_pycorr", "sum"),
                total_logical_errors_my_corr=("total_logical_errors_my_corr", "sum"),
                num_chunks_found=("num_chunks_found", "sum"),
                total_elapsed_seconds=("total_elapsed_seconds", "sum"),
            )
        )
    else:
        print(f"No existing file found. Creating new: {args.outfile}")
        agg_df = new_agg

    # Recompute LERs after combining
    agg_df["ler_pycorr"] = (
        agg_df["total_logical_errors_pycorr"] / agg_df["total_shots"]
    )
    agg_df["ler_my_corr"] = (
        agg_df["total_logical_errors_my_corr"] / agg_df["total_shots"]
    )

    agg_df.to_csv(args.outfile, index=False)
    print(f"Wrote aggregated file to {args.outfile}")


if __name__ == "__main__":
    main()