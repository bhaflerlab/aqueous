#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import anndata as ad


parser = ArgumentParser()
parser.add_argument("--input-h5ad", type=Path, required=True)
parser.add_argument("--output-h5ad", type=Path, required=True)
args = parser.parse_args()

adata = ad.read_h5ad(args.input_h5ad)
if "counts" not in adata.layers:
    raise ValueError("layers['counts'] not found")

args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
adata.X = adata.layers["counts"].copy()
adata.uns["scdrs_input_X"] = "raw_counts_from_layers_counts"
adata.uns["scdrs_input_source_h5ad"] = str(args.input_h5ad)
adata.write_h5ad(args.output_h5ad, compression="gzip")

print(args.output_h5ad)
