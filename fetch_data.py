import os
import shutil
import tarfile
from time import sleep

import bs4
import pandas as pd
import requests
from neurosnap.protein import fetch_uniprot
from tqdm import tqdm

### Functions
# TODO

# prepare download dir
OUTPUT_DIR = "ptms"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR)

# fetch and parse download page
r = requests.get("https://biomics.lab.nycu.edu.tw/dbPTM/download.php")
r.raise_for_status()
soup = bs4.BeautifulSoup(r.text)

# download each ptm file
for el in soup.select("#site .btn.btn-primary"):
  if el.text == "MAC / Linux":
    name = el["href"].split("/")[-1]
    print(f"[+] Fetching {name}")
    r = requests.get("https://biomics.lab.nycu.edu.tw/dbPTM/" + el["href"])
    r.raise_for_status()
    tgz_path = os.path.join(OUTPUT_DIR, name)

    # Save the .tgz file
    with open(tgz_path, "wb") as f:
      f.write(r.content)

    # Extract the .tgz file
    with tarfile.open(tgz_path, "r:gz") as tar:
      tar.extractall(OUTPUT_DIR)

    os.remove(tgz_path)

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