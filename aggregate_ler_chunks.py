import os
import glob
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.indir}")

    dfs = [pd.read_csv(f) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)

    group_cols = ["l", "d", "CD_type", "mem_type", "eta", "p"]

    agg_df = (
        full_df.groupby(group_cols, as_index=False)
        .agg(
            total_shots=("shots_in_chunk", "sum"),
            total_logical_errors_pycorr=("num_logical_errors_pycorr", "sum"),
            total_logical_errors_my_corr=("num_logical_errors_my_corr", "sum"),
            num_chunks_found=("chunk_index", "count"),
            total_elapsed_seconds=("elapsed_seconds", "sum"),
        )
    )

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