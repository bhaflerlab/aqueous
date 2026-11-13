#!/usr/bin/env python3
"""Fit the manuscript CD8 pseudobulk model with continuous age and sex."""

# 1. Read donor-level counts and retain samples with at least five CD8 cells.
from argparse import ArgumentParser
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

parser = ArgumentParser(description=__doc__)
parser.add_argument("--counts", type=Path, required=True)
parser.add_argument("--sample-metadata", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()
design = "~ age_centered + sex + condition"
counts_all = pd.read_csv(args.counts, sep="\t", index_col=0)
metadata = pd.read_csv(args.sample_metadata, sep="\t", index_col="unit_id")
metadata = metadata.loc[metadata.n_cells.ge(5)].copy()
assert metadata.index.is_unique and counts_all.index.is_unique and counts_all.columns.is_unique
assert metadata.Tissue.isin(["aqueous", "pbmc", "csf"]).all()
assert not metadata[["donor_id", "age", "sex"]].isna().any().any()
assert metadata.sex.isin(["F", "M"]).all()
assert np.isfinite(counts_all.to_numpy()).all() and counts_all.to_numpy().min() >= 0
assert np.equal(counts_all.to_numpy(), np.floor(counts_all.to_numpy())).all()
args.output_dir.mkdir(parents=True, exist_ok=False)
runs = []

# 2. Fit each aqueous/reference contrast separately on its eligible genes.
for comparison in ["pbmc", "csf"]:
    samples = metadata.loc[metadata.Tissue.isin(["aqueous", comparison])].copy()
    assert samples.donor_id.is_unique
    assert samples.Tissue.value_counts().reindex(["aqueous", comparison], fill_value=0).ge(3).all()
    samples["condition"] = pd.Categorical(samples.Tissue, categories=[comparison, "aqueous"])
    samples["sex"] = pd.Categorical(samples.sex, categories=["F", "M"])
    samples["age"] = samples.age.astype(float)
    samples["age_centered"] = samples.age - samples.age.mean()
    counts = counts_all.loc[samples.index]
    counts = counts.loc[:, counts.ge(10).sum(axis=0).ge(3)].astype(np.int64)
    inference = DefaultInference(n_cpus=args.threads)
    dds = DeseqDataSet(counts=counts, metadata=samples, design=design,
                      fit_type="parametric", size_factors_fit_type="ratio", refit_cooks=True,
                      min_replicates=7, inference=inference, quiet=True)
    dds.deseq2()
    statistics = DeseqStats(dds, contrast=["condition", "aqueous", comparison], alpha=0.05,
                            cooks_filter=True, independent_filter=True, lfc_null=0,
                            inference=inference, quiet=True)
    statistics.summary()

    # 3. Preserve all estimates and flag genes with nonconverged coefficients.
    result = statistics.results_df.copy()
    result.index.name = "gene"
    for column in ["_genewise_converged", "_MAP_converged", "_LFC_converged"]:
        result[column.removeprefix("_")] = np.asarray(dds.var[column])
    result["inference_eligible"] = result.LFC_converged.eq(1)
    result["significant_fdr05"] = result.inference_eligible & result.padj.lt(0.05)
    result["comparison"] = comparison
    result["design"] = design
    stem = f"261113_cd8_aqueous_vs_{comparison}"
    result.to_csv(args.output_dir / f"{stem}.tsv", sep="\t")
    samples.to_csv(args.output_dir / f"{stem}_samples.tsv", sep="\t")
    runs.append({"comparison": comparison, "samples": samples.Tissue.value_counts().to_dict(),
                 "genes": len(result), "inference_eligible": int(result.inference_eligible.sum()),
                 "significant_fdr05": int(result.significant_fdr05.sum()),
                 "actual_dispersion_trend": dds.uns["disp_function_type"]})
    print(runs[-1], flush=True)

# 4. Record the model settings and software used for these fits.
inputs = []
for path in [Path(__file__), args.counts, args.sample_metadata]:
    with path.open("rb") as handle:
        inputs.append({"path": str(path), "sha256": file_digest(handle, "sha256").hexdigest()})
record = {"status": "completed", "inputs": inputs, "design": design, "runs": runs,
          "gene_filter": "Raw count >=10 in >=3 samples within each comparison",
          "sample_filter": "At least 5 CD8 cells per donor/tissue unit",
          "age_centering": "Subtract the mean age of samples within each comparison",
          "inference": "Unshrunk two-sided condition Wald test; default Cook and independent filtering",
          "software": {name: version(name) for name in ["pydeseq2", "numpy", "pandas", "scipy", "anndata"]}}
(args.output_dir / "261113_cd8_pseudobulk.json").write_text(json.dumps(record, indent=2) + "\n")
