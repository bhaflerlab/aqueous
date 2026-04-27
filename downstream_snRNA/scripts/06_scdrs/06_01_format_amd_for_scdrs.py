#!/usr/bin/env python3

from argparse import ArgumentParser
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def chunk_to_scdrs(df: pd.DataFrame) -> pd.DataFrame:
    p = pd.to_numeric(df["GC.Pvalue"], errors="coerce").clip(lower=np.nextafter(0, 1), upper=1.0)
    sign = df["Overall"].map({"+": 1.0, "-": -1.0})
    z = sign * norm.isf(p / 2.0)

    return pd.DataFrame(
        {
            "SNP": df["Marker"].astype(str),
            "CHR": df["Chrom"].astype(str),
            "BP": pd.to_numeric(df["Pos"], errors="coerce").astype("Int64"),
            "A1": df["Allele1"].astype(str),
            "A2": df["Allele2"].astype(str),
            "OR": pd.Series(pd.NA, index=df.index, dtype="object"),
            "BETA": pd.Series(pd.NA, index=df.index, dtype="object"),
            "SE": pd.Series(pd.NA, index=df.index, dtype="object"),
            "Z": z,
            "P": p,
            "EAF": pd.Series(pd.NA, index=df.index, dtype="object"),
            "N": pd.to_numeric(df["Ncases"], errors="coerce") + pd.to_numeric(df["Ncontrols"], errors="coerce"),
        }
    )


parser = ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)
reader = pd.read_csv(args.input, sep="\t", chunksize=1_000_000)
with gzip.open(args.output, "wt") as fout:
    first = True
    for chunk in reader:
        out = chunk_to_scdrs(chunk)
        out.to_csv(fout, sep="\t", index=False, header=first)
        first = False

print(args.output)
