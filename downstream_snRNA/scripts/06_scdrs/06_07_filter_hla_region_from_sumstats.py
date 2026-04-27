#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


HLA_CHR = "6"
HLA_START = 25_000_000
HLA_END = 34_000_000


parser = ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t", low_memory=False)
df["CHR"] = df["CHR"].astype(str).str.replace("^chr", "", regex=True)
df["BP"] = pd.to_numeric(df["BP"], errors="coerce")

mask_hla = (df["CHR"] == HLA_CHR) & (df["BP"] >= HLA_START) & (df["BP"] <= HLA_END)
df_out = df.loc[~mask_hla].copy()

args.output.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(args.output, sep="\t", index=False, compression="gzip")
pd.DataFrame(
    [
        {"metric": "input_path", "value": str(args.input)},
        {"metric": "output_path", "value": str(args.output)},
        {"metric": "input_rows", "value": int(df.shape[0])},
        {"metric": "removed_hla_rows", "value": int(mask_hla.sum())},
        {"metric": "output_rows", "value": int(df_out.shape[0])},
        {"metric": "hla_chr", "value": HLA_CHR},
        {"metric": "hla_start", "value": HLA_START},
        {"metric": "hla_end", "value": HLA_END},
    ]
).to_csv(args.report, sep="\t", index=False)

print(args.output)
print(args.report)
