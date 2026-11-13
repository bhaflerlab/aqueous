#!/usr/bin/env python3
"""Run original and direct postmortem-gene exclusion Hallmark GSEA."""

# 1. Read the external whole-blood effects and the Hallmark collection.
from argparse import ArgumentParser
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
import json

import gseapy as gp
import numpy as np
import pandas as pd

parser = ArgumentParser(description=__doc__)
parser.add_argument("--deg-dir", type=Path, required=True)
parser.add_argument("--ferreira-supplement", type=Path, required=True)
parser.add_argument("--hallmark-gmt", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()
design = "~ age_centered + sex + condition"
terms = ["HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_HYPOXIA", "HALLMARK_APOPTOSIS"]
columns = ["Term", "ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val", "Tag %", "Gene %", "Lead_genes"]
external = pd.read_excel(args.ferreira_supplement, sheet_name="DiffExp_Blood_Pre_vs_Postm", header=1)
nontext = ~external.Description.map(lambda value: isinstance(value, str))
symbol_audit = external.loc[nontext].assign(exclusion_reason="nontext_symbol")
external = external.loc[~nontext].copy()
ambiguous = external.Description.duplicated(keep=False)
symbol_audit = pd.concat([symbol_audit, external.loc[ambiguous].assign(exclusion_reason="ambiguous_symbol")])
external = external.loc[~ambiguous].copy()
strict = external.FDR.lt(0.01) & external.logFC.abs().ge(1)
exclusion_genes = set(external.loc[strict, "Description"])
gene_sets = {}
for line in args.hallmark_gmt.read_text().splitlines():
    term, description, *genes = line.split("\t")
    gene_sets[term] = genes
args.output_dir.mkdir(parents=True, exist_ok=False)
symbol_audit.to_csv(args.output_dir / "261113_ferreira_symbol_exclusions.tsv", sep="\t", index=False)
external.loc[strict, ["Description", "Name", "logFC", "FDR"]].sort_values("Description").to_csv(
    args.output_dir / "261113_ferreira_exclusion_genes.tsv", sep="\t", index=False)
input_paths = [Path(__file__), args.ferreira_supplement, args.hallmark_gmt]
all_results, runs = [], []

# 2. Build the complete model-eligible signed Wald rankings without DEG filtering.
for comparison in ["pbmc", "csf"]:
    deg_path = args.deg_dir / f"261113_cd8_aqueous_vs_{comparison}.tsv"
    input_paths.append(deg_path)
    deg = pd.read_csv(deg_path, sep="\t", float_precision="round_trip")
    assert deg.gene.is_unique and deg.design.eq(design).all()
    eligible = deg.inference_eligible & np.isfinite(deg.stat) & np.isfinite(deg.pvalue)
    original = deg.loc[eligible, ["gene", "stat"]].sort_values(
        ["stat", "gene"], ascending=[False, True]).reset_index(drop=True)
    removed = original.gene.isin(exclusion_genes)
    excluded = original.loc[~removed].reset_index(drop=True)
    original.loc[removed].to_csv(args.output_dir / f"261113_cd8_{comparison}_removed_genes.tsv", sep="\t", index=False)

    # 3. Remove only the external genes; retain genes untested by Ferreira et al.
    for stage, ranks in [("original", original), ("excluded", excluded)]:
        ranks.to_csv(args.output_dir / f"261113_cd8_{comparison}_{stage}_ranking.tsv", sep="\t", index=False)
        fit = gp.prerank(rnk=ranks.copy(), gene_sets=gene_sets, outdir=None,
                         min_size=15, max_size=500, permutation_num=1000, weight=1,
                         threads=args.threads, seed=1729, no_plot=True, verbose=False)
        assert list(fit.ranking.index) == list(ranks.gene)
        np.testing.assert_array_equal(fit.ranking.to_numpy(), ranks.stat.to_numpy())
        result = fit.res2d[columns].copy()
        result[columns[1:6]] = result[columns[1:6]].apply(pd.to_numeric)
        result["comparison"], result["stage"] = comparison, stage
        result["ranked_genes"], result["design"] = len(ranks), design
        result.to_csv(args.output_dir / f"261113_cd8_{comparison}_{stage}_hallmark.tsv", sep="\t", index=False)
        all_results.append(result)
        runs.append({"comparison": comparison, "stage": stage, "ranked_genes": len(ranks),
                     "tested_sets": len(result), "removed_genes": 0 if stage == "original" else int(removed.sum())})
        print(runs[-1], flush=True)
        del fit

# 4. Save the selected pathways and provenance for the two paired comparisons.
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(args.output_dir / "261113_cd8_all_hallmark.tsv", sep="\t", index=False)
combined.loc[combined.Term.isin(terms)].to_csv(
    args.output_dir / "261113_cd8_postmortem_selected_pathways.tsv", sep="\t", index=False)
inputs = []
for path in input_paths:
    with path.open("rb") as handle:
        inputs.append({"path": str(path), "sha256": file_digest(handle, "sha256").hexdigest()})
record = {"status": "completed", "inputs": inputs, "design": design, "runs": runs,
          "external_source": "Ferreira et al. 2018 Supplementary Data 2; whole-blood pre/postmortem comparison",
          "external_definition": "FDR <0.01 and absolute log2FC >=1; exact unambiguous text symbols",
          "external_gene_count": len(exclusion_genes), "common_gene_restriction": False,
          "ranking": "Unshrunk condition Wald statistic; inference_eligible and finite stat/pvalue; no padj filter",
          "permutations": 1000, "seed": 1729, "weight": 1, "min_size": 15, "max_size": 500,
          "fdr_scope": "GSEA gene-set permutation FDR within eligible Hallmark sets per ranking",
          "zero_q_note": "A finite-permutation estimate, not an exact zero error probability",
          "software": {name: version(name) for name in ["gseapy", "numpy", "pandas", "openpyxl"]}}
(args.output_dir / "261113_cd8_postmortem_gsea.json").write_text(json.dumps(record, indent=2) + "\n")
