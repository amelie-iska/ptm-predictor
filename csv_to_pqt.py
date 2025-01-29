#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script: data_preprocessing.py

Reads CSV files from the 'ptms/' directory, each containing:
    - name
    - uniprot
    - ptm_location (position of the PTM in the sequence)
    - ptm_name (the type of PTM)
    - unknown_stupid
    - adjacent_seq
    - seq (the full protein sequence)

For each row:
  1) Build a label array for each amino acid in seq (multi-class or single-class).
  2) Chunk the sequence + label array into smaller pieces to prevent large memory usage.
  3) Write out the chunks to multiple Parquet files.
"""

import os
import math
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from glob import glob

# ------------- CONFIGURATION -------------
CHUNK_SIZE = 512     # Max tokens per chunk
OVERLAP = 0          # Overlap between chunks (set >0 if desired)
OUTPUT_DIR = "chunks"  # Where we'll store parquet shards
# Example: define a small mapping from PTM names to numeric class IDs.
# In practice, you might have a large dictionary for 72+ PTMs.
PTM_LABEL2ID = {
    "phosphorylation": 1,
    "glycosylation": 2,
    "acetylation": 3,
    "methylation": 4,
    "UNKNOWN_PTM": 5,
    "NONE": 0,  # For positions without PTMs
}

# ------------- HELPER FUNCTIONS -------------
def build_label_array(seq, ptm_location, ptm_name):
    """
    Build an integer label array (same length as seq).
    For multi-class classification of PTM types:
      - Mark the PTM position with the label ID for that PTM type.
      - If multiple PTMs exist, you'll need a more complex approach 
        (like storing the max or storing them in separate data rows).
    """
    labels = [PTM_LABEL2ID["NONE"]] * len(seq)

    # Convert PTM location from (1-based) to (0-based) index:
    # e.g., if ptm_location is '123', this means the 123rd AA in seq.
    # Watch out for possible multiple sites in one row, not shown here.
    try:
        location_idx = int(ptm_location) - 1
        if 0 <= location_idx < len(seq):
            ptm_type_id = PTM_LABEL2ID.get(ptm_name, PTM_LABEL2ID["UNKNOWN_PTM"])
            labels[location_idx] = ptm_type_id
    except (ValueError, TypeError):
        pass  # if parsing fails or missing data, default to "NONE"

    return labels


def chunk_sequence_and_labels(seq, labels, chunk_size=512, overlap=0):
    """
    Generator that yields (seq_chunk, label_chunk) pairs.
    Each chunk has length <= chunk_size.
    Overlap is optional.
    """
    start = 0
    while start < len(seq):
        end = start + chunk_size
        seq_chunk = seq[start:end]
        label_chunk = labels[start:end]
        yield seq_chunk, label_chunk
        start += (chunk_size - overlap)  # advance by chunk_size - overlap


# ------------- MAIN PREPROCESSING -------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = glob(os.path.join("ptms", "*.csv"))
    if not csv_files:
        print("No CSV files found in 'ptms/'. Exiting.")
        return

    # We'll collect chunked data in memory, then write out in multiple Parquet shards
    # for streaming. Alternatively, you can write each chunk on the fly.
    data_rows = []

    for csv_path in csv_files:
        df = pd.read_csv(
            csv_path,
            sep="\t",
            names=["name", "uniprot", "ptm_location", "ptm_name", "unknown_stupid", "adjacent_seq", "seq"],
            header=None
        )
        print(f"[+] Processing {csv_path}, {len(df)} rows.")

        for _, row in df.iterrows():
            seq = str(row["seq"])
            ptm_location = row["ptm_location"]
            ptm_name = str(row["ptm_name"])

            # Build label array for the entire sequence
            labels = build_label_array(seq, ptm_location, ptm_name)

            # Chunk the seq and labels
            for seq_chunk, label_chunk in chunk_sequence_and_labels(seq, labels, CHUNK_SIZE, OVERLAP):
                # Store a row with just the chunked seq and label
                data_rows.append({
                    "uniprot": row["uniprot"],
                    "ptm_name": ptm_name,
                    "chunked_seq": seq_chunk,
                    "chunked_labels": label_chunk
                })

    # Convert to a DataFrame
    final_df = pd.DataFrame(data_rows)
    total_rows = len(final_df)
    print(f"[+] Total chunked rows: {total_rows}")

    # Write out to multiple Parquet files (shards). 
    # We'll pick a shard size, e.g., 50k rows per shard.
    shard_size = 50000
    num_shards = math.ceil(total_rows / shard_size)

    for shard_id in range(num_shards):
        start_idx = shard_id * shard_size
        end_idx = min((shard_id + 1) * shard_size, total_rows)
        shard_df = final_df.iloc[start_idx:end_idx]

        shard_path = os.path.join(OUTPUT_DIR, f"chunks_{shard_id:03d}.parquet")
        print(f"    -> Writing {shard_path} with rows [{start_idx}, {end_idx})")
        shard_table = pa.Table.from_pandas(shard_df)
        pq.write_table(shard_table, shard_path)

    print("[+] Done. Parquet shards saved in 'chunks/'.")


if __name__ == "__main__":
    main()