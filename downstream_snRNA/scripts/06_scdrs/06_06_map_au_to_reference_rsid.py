#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def load_bim_map(bim_path: Path) -> pd.DataFrame:
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1_REF", "A2_REF"],
        low_memory=False,
    )
    bim["CHR"] = bim["CHR"].astype(str)
    bim["BP"] = pd.to_numeric(bim["BP"], errors="coerce")
    bim["A_LOW"] = bim[["A1_REF", "A2_REF"]].astype(str).apply(lambda x: min(x[0], x[1]), axis=1)
    bim["A_HIGH"] = bim[["A1_REF", "A2_REF"]].astype(str).apply(lambda x: max(x[0], x[1]), axis=1)
    return bim[["CHR", "BP", "A_LOW", "A_HIGH", "SNP"]].drop_duplicates()


parser = ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--bim", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t", low_memory=False)
df["CHR"] = df["CHR"].astype(str).str.replace("^chr", "", regex=True)
df["BP"] = pd.to_numeric(df["BP"], errors="coerce")
df["A1"] = df["A1"].astype(str)
df["A2"] = df["A2"].astype(str)
df["A_LOW"] = df[["A1", "A2"]].apply(lambda x: min(x[0], x[1]), axis=1)
df["A_HIGH"] = df[["A1", "A2"]].apply(lambda x: max(x[0], x[1]), axis=1)

bim = load_bim_map(args.bim)
df_out = df.merge(bim, on=["CHR", "BP", "A_LOW", "A_HIGH"], how="left")

n_input = int(df_out.shape[0])
n_mapped = int(df_out["SNP_y"].notna().sum())
n_unmapped = n_input - n_mapped

df_out = df_out[df_out["SNP_y"].notna()].copy()
df_out["SNP"] = df_out["SNP_y"].astype(str)
keep_cols = ["SNP", "CHR", "BP", "A1", "A2", "OR", "BETA", "SE", "Z", "P", "EAF", "N"]

args.output.parent.mkdir(parents=True, exist_ok=True)
df_out[keep_cols].to_csv(args.output, sep="\t", index=False, compression="gzip")
pd.DataFrame(
    {
        "input_file": [str(args.input)],
        "bim_file": [str(args.bim)],
        "output_file": [str(args.output)],
        "n_input": [n_input],
        "n_mapped": [n_mapped],
        "n_unmapped": [n_unmapped],
        "mapping_rate": [n_mapped / n_input if n_input > 0 else 0.0],
    }
).to_csv(args.report, sep="\t", index=False)

print(args.output)
print(args.report)
