"""
Diagnostic script to verify embedding quality.
Checks: dimensions, variance, constant columns, NaN/Inf values.

Usage:
    python diagnose_embeddings.py --embedding_file processed_datasets/embeddings_citiesllama2_7b_1_rmv_period.csv
    python diagnose_embeddings.py --embedding_dir processed_datasets --pattern "embeddings_*llama2*"
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import glob


def correct_str(str_arr):
    val_to_ret = (str_arr.replace("[array(", "")
                        .replace("dtype=float32)]", "")
                        .replace("\n", "")
                        .replace(" ", "")
                        .replace("],", "]")
                        .replace("[", "")
                        .replace("]", ""))
    return val_to_ret


def diagnose_file(filepath):
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if 'embeddings' not in df.columns:
        print("ERROR: No 'embeddings' column found!")
        return

    # Parse embeddings
    raw_embeddings = df['embeddings'].tolist()
    embeddings_list = []
    parse_errors = 0
    for i, emb_str in enumerate(raw_embeddings):
        try:
            parsed = np.fromstring(correct_str(str(emb_str)), sep=',')
            embeddings_list.append(parsed)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:
                print(f"  Parse error at row {i}: {e}")

    if parse_errors > 0:
        print(f"WARNING: {parse_errors} rows failed to parse")

    if not embeddings_list:
        print("ERROR: No valid embeddings found!")
        return

    embeddings = np.array(embeddings_list)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    # Check for NaN/Inf
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")

    # Check variance
    per_dim_var = np.var(embeddings, axis=0)
    zero_var_dims = np.sum(per_dim_var == 0)
    low_var_dims = np.sum(per_dim_var < 1e-10)
    print(f"Total dimensions: {len(per_dim_var)}")
    print(f"Zero-variance dimensions: {zero_var_dims}")
    print(f"Low-variance dimensions (<1e-10): {low_var_dims}")
    print(f"Mean variance: {per_dim_var.mean():.6e}")
    print(f"Min variance: {per_dim_var.min():.6e}")
    print(f"Max variance: {per_dim_var.max():.6e}")

    # Overall embedding statistics
    print(f"\nEmbedding value stats:")
    print(f"  Mean: {np.nanmean(embeddings):.6e}")
    print(f"  Std:  {np.nanstd(embeddings):.6e}")
    print(f"  Min:  {np.nanmin(embeddings):.6e}")
    print(f"  Max:  {np.nanmax(embeddings):.6e}")

    # Check if embeddings are all the same
    first_emb = embeddings[0]
    all_same = np.allclose(embeddings, first_emb, atol=1e-6)
    print(f"\nAll embeddings nearly identical (atol=1e-6): {all_same}")

    # Per-sample norm
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding L2 norms:")
    print(f"  Mean: {norms.mean():.6e}")
    print(f"  Std:  {norms.std():.6e}")
    print(f"  Min:  {norms.min():.6e}")
    print(f"  Max:  {norms.max():.6e}")

    # Label distribution
    if 'label' in df.columns:
        labels = df['label'].values
        print(f"\nLabel distribution:")
        print(f"  True (1): {np.sum(labels == 1)}")
        print(f"  False (0): {np.sum(labels == 0)}")

        # Check if embeddings differ between true/false
        true_emb = embeddings[labels == 1]
        false_emb = embeddings[labels == 0]
        true_mean = np.nanmean(true_emb, axis=0)
        false_mean = np.nanmean(false_emb, axis=0)
        diff = np.linalg.norm(true_mean - false_mean)
        print(f"  L2 distance between true/false mean embeddings: {diff:.6e}")

    # Show first few embedding values
    print(f"\nFirst embedding (first 10 values): {embeddings[0][:10]}")
    print(f"Last embedding (first 10 values): {embeddings[-1][:10]}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose embedding quality.")
    parser.add_argument("--embedding_file", help="Path to a single embedding CSV file.")
    parser.add_argument("--embedding_dir", help="Directory containing embedding CSV files.")
    parser.add_argument("--pattern", default="embeddings_*.csv", help="Glob pattern for finding files in --embedding_dir.")
    args = parser.parse_args()

    if args.embedding_file:
        diagnose_file(args.embedding_file)
    elif args.embedding_dir:
        files = sorted(glob.glob(str(Path(args.embedding_dir) / args.pattern)))
        if not files:
            print(f"No files matching '{args.pattern}' found in {args.embedding_dir}")
            return
        for f in files:
            diagnose_file(f)
    else:
        print("Please provide --embedding_file or --embedding_dir")


if __name__ == "__main__":
    main()
