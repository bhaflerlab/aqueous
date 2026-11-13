#!/usr/bin/env python3
"""Sum raw CD8 counts within the curated donor/tissue sample units."""

# 1. Read the annotated counts and the curated sample roster.
from argparse import ArgumentParser
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse

parser = ArgumentParser(description=__doc__)
parser.add_argument("--input-h5ad", type=Path, required=True)
parser.add_argument("--sample-metadata", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--label-col", default="predicted_labels_Immune_All_Low_leiden_r05")
args = parser.parse_args()
labels = ["Tem/Trm cytotoxic T cells", "Tem/Temra cytotoxic T cells",
          "Tcm/Naive cytotoxic T cells", "Trm cytotoxic T cells"]
metadata = pd.read_csv(args.sample_metadata, sep="\t", index_col="unit_id")
metadata = metadata[["Tissue", "sample", "SampleID", "donor_id", "age", "sex"]].copy()
assert metadata.index.is_unique
assert metadata.Tissue.isin(["aqueous", "pbmc", "csf"]).all()
assert not metadata.loc[metadata.Tissue.eq("aqueous"), "sample"].str.contains("hashed", case=False).any()
with h5py.File(args.input_h5ad, "r") as handle:
    obs = read_elem(handle["obs"])
    var = read_elem(handle["var"])
    counts = sparse.csr_matrix(read_elem(handle["layers/counts"]))
assert obs.index.is_unique and var.index.is_unique
assert np.isfinite(counts.data).all() and (counts.data >= 0).all()
assert np.equal(counts.data, np.floor(counts.data)).all()
selected = obs[args.label_col].isin(labels)

# 2. Aggregate each sample using its original aqueous or reference identifier.
pseudobulk = pd.DataFrame(0, index=metadata.index, columns=var.index, dtype=np.int64)
cell_units = pd.Series("", index=obs.index, name="unit_id")
for unit_id, sample in metadata.iterrows():
    source_column = "sample" if sample.Tissue == "aqueous" else "SampleID"
    assert pd.notna(sample[source_column])
    mask = selected & obs.Tissue.eq(sample.Tissue) & obs[source_column].eq(sample[source_column])
    assert cell_units.loc[mask].eq("").all(), "A cell cannot belong to two sample units."
    cell_units.loc[mask] = unit_id
    metadata.loc[unit_id, "n_cells"] = int(mask.sum())
    pseudobulk.loc[unit_id] = np.asarray(counts[mask.to_numpy()].sum(axis=0)).ravel().astype(np.int64)
metadata["n_cells"] = metadata.n_cells.astype(int)

# 3. Save count sums, sample metadata, and a record of the input sources.
args.output_dir.mkdir(parents=True, exist_ok=False)
pseudobulk.to_csv(args.output_dir / "261113_cd8_counts.tsv.gz", sep="\t")
metadata.to_csv(args.output_dir / "261113_cd8_sample_metadata.tsv", sep="\t")
obs.loc[cell_units.ne(""), ["Tissue", args.label_col]].join(cell_units).to_csv(
    args.output_dir / "261113_cd8_cell_units.tsv.gz", sep="\t", index_label="cell_id")
inputs = []
for path in [Path(__file__), args.input_h5ad, args.sample_metadata]:
    with path.open("rb") as handle:
        inputs.append({"path": str(path), "sha256": file_digest(handle, "sha256").hexdigest()})
record = {"status": "completed", "inputs": inputs, "labels": labels,
          "count_source": "layers/counts; integer raw-count sums; no normalization",
          "sample_units": len(metadata), "selected_cells": int(metadata.n_cells.sum()),
          "software": {name: version(name) for name in ["anndata", "h5py", "numpy", "pandas", "scipy"]}}
(args.output_dir / "261113_cd8_preparation.json").write_text(json.dumps(record, indent=2) + "\n")
print(metadata.groupby("Tissue").agg(samples=("n_cells", "size"), cells=("n_cells", "sum")))
