import os
from time import sleep

import pandas as pd
from neurosnap.protein import fetch_uniprot
from tqdm import tqdm

# add full uniprot sequences to each dataframe
for fname in os.listdir("ptms"):
  fpath = os.path.join("ptms", fname)
  df = pd.read_csv(fpath, sep="\t", names=["name", "uniprot", "ptm_location", "ptm_name", "unknown_stupid", "adjacent_seq"])
  full_seqs = []
  for _, row in tqdm(df.iterrows()):
    for _ in range(50):
      try:
        full_seqs.append(fetch_uniprot(row.uniprot))
        break
      except:
        sleep(5)
  df["seq"] = full_seqs
  df.to_csv(fpath, index=False)