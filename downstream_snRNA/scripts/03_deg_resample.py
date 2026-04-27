#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse as sp
from scipy.stats import chi2, mannwhitneyu
from statsmodels.stats.multitest import multipletests


analysis_configs = {
    "CD8_4labels": {
        "labels": [
            "Tem/Trm cytotoxic T cells",
            "Tem/Temra cytotoxic T cells",
            "Tcm/Naive cytotoxic T cells",
            "Trm cytotoxic T cells",
        ],
        "tissue_order": ["aqueous", "pbmc", "csf"],
    },
    "Monocyte_like_3labels": {
        "labels": [
            "Mono-mac",
            "Classical monocytes",
            "Non-classical monocytes",
        ],
        "tissue_order": ["aqueous", "pbmc", "csf"],
    },
    "Monocyte_2labels": {
        "labels": [
            "Classical monocytes",
            "Non-classical monocytes",
        ],
        "tissue_order": ["aqueous", "pbmc", "csf"],
    },
}


def get_dense_matrix(X, idx):
    if sp.issparse(X):
        return X[idx, :].toarray()
    return np.asarray(X[idx, :])


def de_one_resample(adata, idx_aq, idx_other, n_per_group, seed=1, min_pct=0.1):
    rng = np.random.default_rng(seed)

    aq_take = idx_aq if len(idx_aq) == n_per_group else rng.choice(idx_aq, n_per_group, replace=False)
    ot_take = rng.choice(idx_other, n_per_group, replace=False)

    Xa = get_dense_matrix(adata.X, adata.obs_names.get_indexer(aq_take))
    Xo = get_dense_matrix(adata.X, adata.obs_names.get_indexer(ot_take))

    pct_a = (Xa > 0).mean(axis=0)
    pct_o = (Xo > 0).mean(axis=0)
    keep = (pct_a >= min_pct) | (pct_o >= min_pct)

    mu_a = Xa.mean(axis=0)
    mu_o = Xo.mean(axis=0)
    expr_diff = mu_a - mu_o

    pvals = np.full(adata.n_vars, np.nan, dtype=float)
    kept_idx = np.where(keep)[0]
    for j in kept_idx:
        _, p = mannwhitneyu(Xa[:, j], Xo[:, j], alternative="two-sided")
        pvals[j] = p

    out = pd.DataFrame(
        {
            "gene": adata.var_names,
            "expr_diff": expr_diff,
            "pct.1": pct_a,
            "pct.2": pct_o,
            "p_val": pvals,
            "seed": seed,
        }
    )
    return out.loc[keep].reset_index(drop=True)


def combine_p_fisher(pvals):
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    p = np.clip(p, 1e-300, 1.0)
    stat = -2.0 * np.sum(np.log(p))
    return chi2.sf(stat, df=2 * p.size)


def deg_resample(adata, tissue_col, tissue_other, seeds, min_pct=0.1):
    idx_aq = adata.obs_names[adata.obs[tissue_col] == "aqueous"].to_numpy()
    idx_ot = adata.obs_names[adata.obs[tissue_col] == tissue_other].to_numpy()
    n_per_group = len(idx_aq)

    if len(idx_aq) == 0:
        raise ValueError("No aqueous cells found in subset.")
    if len(idx_ot) < n_per_group:
        raise ValueError(f"Not enough cells in {tissue_other}: {len(idx_ot)} < {n_per_group}")

    res = [
        de_one_resample(
            adata=adata,
            idx_aq=idx_aq,
            idx_other=idx_ot,
            n_per_group=n_per_group,
            seed=sd,
            min_pct=min_pct,
        )
        for sd in seeds
    ]

    long = pd.concat(res, axis=0, ignore_index=True)
    grp = long.groupby("gene", sort=False)
    deg = grp.agg(
        expr_diff_mean=("expr_diff", "mean"),
        pct1_mean=("pct.1", "mean"),
        pct2_mean=("pct.2", "mean"),
        p_fisher=("p_val", combine_p_fisher),
    ).reset_index()

    p = deg["p_fisher"].to_numpy()
    p_adj = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() > 0:
        p_adj[ok] = multipletests(p[ok], method="fdr_bh")[1]
    deg["p_adj_fisher"] = p_adj
    return deg.sort_values("p_adj_fisher", na_position="last").reset_index(drop=True), long


parser = ArgumentParser()
parser.add_argument("--input-h5ad", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--fig-dir", type=Path, required=True)
parser.add_argument("--group", required=True, choices=list(analysis_configs.keys()))
parser.add_argument("--label-col", default="predicted_labels_Immune_All_Low_leiden_r05")
parser.add_argument("--tissue-col", default="Tissue")
parser.add_argument("--seeds", type=int, default=100)
parser.add_argument("--min-pct", type=float, default=0.1)
parser.add_argument("--target-sum", type=float, default=1e4)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
args.fig_dir.mkdir(parents=True, exist_ok=True)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

cfg = analysis_configs[args.group]
print(f"Reading: {args.input_h5ad}")
adata = sc.read_h5ad(args.input_h5ad)
print(adata)

adata_sub = adata[adata.obs[args.label_col].astype("object").isin(cfg["labels"])].copy()

count_tab = pd.crosstab(adata_sub.obs[args.tissue_col], adata_sub.obs[args.label_col]).reindex(
    index=cfg["tissue_order"],
    columns=cfg["labels"],
    fill_value=0,
)
print("Counts by tissue:")
print(count_tab)

if "counts" not in adata_sub.layers:
    raise ValueError("counts layer is required for DEG. Do not reuse adata.X when its provenance is unclear.")

adata_sub.X = adata_sub.layers["counts"].copy()
sc.pp.normalize_total(adata_sub, target_sum=args.target_sum)
sc.pp.log1p(adata_sub)

seeds = range(1, args.seeds + 1)
for tissue_other in ["pbmc", "csf"]:
    print(f"Running {args.group}: aqueous vs {tissue_other}")
    deg_df, long_df = deg_resample(
        adata=adata_sub,
        tissue_col=args.tissue_col,
        tissue_other=tissue_other,
        seeds=seeds,
        min_pct=args.min_pct,
    )

    deg_out = args.output_dir / f"DEG_{args.group}_aqueous_vs_{tissue_other}_log1p_resamp{args.seeds}_v01.tsv"
    long_out = args.output_dir / f"DEGlong_{args.group}_aqueous_vs_{tissue_other}_log1p_resamp{args.seeds}_v01.tsv.gz"

    if deg_out.exists() or long_out.exists():
        raise FileExistsError(f"Output already exists in {args.output_dir}")

    deg_df.to_csv(deg_out, sep="\t", index=False)
    long_df.to_csv(long_out, sep="\t", index=False, compression="gzip")
    print(f"Saved: {deg_out}")
    print(f"Saved: {long_out}")

    plot_df = deg_df.copy()
    plot_df["neglog10_fdr"] = -np.log10(plot_df["p_adj_fisher"].clip(lower=1e-300))
    plot_df["direction"] = "Below_cutoff"
    up_mask = (plot_df["p_adj_fisher"] < 0.05) & (plot_df["expr_diff_mean"] >= 0.25)
    down_mask = (plot_df["p_adj_fisher"] < 0.05) & (plot_df["expr_diff_mean"] <= -0.25)
    plot_df.loc[up_mask, "direction"] = "Aqueous_up"
    plot_df.loc[down_mask, "direction"] = "Aqueous_down"

    lab_df = pd.concat(
        [
            plot_df.loc[up_mask].sort_values("expr_diff_mean", ascending=False).head(12),
            plot_df.loc[down_mask].sort_values("expr_diff_mean", ascending=True).head(12),
        ],
        axis=0,
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    sns.scatterplot(
        data=plot_df,
        x="expr_diff_mean",
        y="neglog10_fdr",
        hue="direction",
        palette={
            "Aqueous_up": "#D62728",
            "Aqueous_down": "#1F77B4",
            "Below_cutoff": "#4D4D4D",
        },
        s=16,
        linewidth=0,
        alpha=0.8,
        ax=ax,
    )
    ax.axvline(-0.25, linestyle=":", linewidth=1, color="0.5")
    ax.axvline(0.25, linestyle=":", linewidth=1, color="0.5")
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1, color="0.5")
    for _, row in lab_df.iterrows():
        ax.text(
            row["expr_diff_mean"],
            row["neglog10_fdr"] + 0.08,
            row["gene"],
            fontsize=7,
            ha="center",
        )
    ax.set_xlabel("Mean normalized log-expression difference across resamples")
    ax.set_ylabel("-log10(FDR) [Fisher + BH]")
    ax.set_title(f"{args.group}: aqueous vs {tissue_other}")
    ax.grid(False)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        args.fig_dir / f"Volcano_{args.group}_aqueous_vs_{tissue_other}_resamp{args.seeds}_v01.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
