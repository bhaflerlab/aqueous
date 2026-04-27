#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


N_TOTAL = 920399


parser = ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t", low_memory=False)
p = pd.to_numeric(df["p_value"], errors="coerce").clip(lower=np.nextafter(0, 1), upper=1.0)
or_val = pd.to_numeric(df["odds_ratio"], errors="coerce")
se = pd.to_numeric(df["standard_error"], errors="coerce")
beta = np.log(or_val)
z = beta / se

out = pd.DataFrame(
    {
        "SNP": df["Name"].astype(str),
        "CHR": df["chromosome"].astype(str),
        "BP": pd.to_numeric(df["base_pair_location"], errors="coerce").astype("Int64"),
        "A1": df["effect_allele"].astype(str),
        "A2": df["other_allele"].astype(str),
        "OR": or_val,
        "BETA": beta,
        "SE": se,
        "Z": z,
        "P": p,
        "EAF": pd.to_numeric(df["effect_allele_frequency"], errors="coerce"),
        "N": N_TOTAL,
    }
)
out = out.dropna(subset=["SNP", "CHR", "BP", "A1", "A2", "P"]).reset_index(drop=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, sep="\t", index=False, compression="gzip")
pd.DataFrame(
    [
        {"metric": "input_file", "value": str(args.input)},
        {"metric": "output_file", "value": str(args.output)},
        {"metric": "input_rows", "value": int(df.shape[0])},
        {"metric": "output_rows", "value": int(out.shape[0])},
        {"metric": "n_total", "value": N_TOTAL},
    ]
).to_csv(args.report, sep="\t", index=False)

print(args.output)
print(args.report)
