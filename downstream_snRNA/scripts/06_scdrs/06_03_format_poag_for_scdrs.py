#!/usr/bin/env python3

from argparse import ArgumentParser
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def make_snp_id(df: pd.DataFrame) -> pd.Series:
    rs = df["rsids"].astype(str)
    fallback = (
        df["#chrom"].astype(str)
        + ":"
        + df["pos"].astype(str)
        + ":"
        + df["ref"].astype(str)
        + ":"
        + df["alt"].astype(str)
    )
    return rs.where((rs != "") & (rs != ".") & (rs != "nan"), fallback)


def chunk_to_scdrs(df: pd.DataFrame, total_n: int) -> pd.DataFrame:
    beta = pd.to_numeric(df["beta"], errors="coerce")
    se = pd.to_numeric(df["sebeta"], errors="coerce")
    p = pd.to_numeric(df["pval"], errors="coerce").clip(lower=np.nextafter(0, 1), upper=1.0)
    return pd.DataFrame(
        {
            "SNP": make_snp_id(df),
            "CHR": df["#chrom"].astype(str),
            "BP": pd.to_numeric(df["pos"], errors="coerce").astype("Int64"),
            "A1": df["alt"].astype(str),
            "A2": df["ref"].astype(str),
            "OR": np.exp(beta),
            "BETA": beta,
            "SE": se,
            "Z": beta / se,
            "P": p,
            "EAF": pd.to_numeric(df["af_alt"], errors="coerce"),
            "N": total_n,
        }
    )


parser = ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--n-total", type=int, default=7756 + 358375)
args = parser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)
reader = pd.read_csv(args.input, sep="\t", chunksize=1_000_000)
with gzip.open(args.output, "wt") as fout:
    first = True
    for chunk in reader:
        out = chunk_to_scdrs(chunk, args.n_total)
        out.to_csv(fout, sep="\t", index=False, header=first)
        first = False

print(args.output)
