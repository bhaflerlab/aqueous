#!/usr/bin/env python3

from argparse import ArgumentParser
import gc
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import scipy.sparse as sp
import seaborn as sns


SELECTED_SUBTYPE_LABELS_REDUCED = [
    "Tem/Trm cytotoxic T cells",
    "Tem/Temra cytotoxic T cells",
    "Tcm/Naive cytotoxic T cells",
    "Trm cytotoxic T cells",
    "Tcm/Naive helper T cells",
    "Tem/Effector helper T cells",
    "Regulatory T cells",
    "CD16- NK cells",
    "CD16+ NK cells",
    "Naive B cells",
    "Memory B cells",
    "Plasma cells",
    "Classical monocytes",
    "Non-classical monocytes",
    "Macrophages",
]

SELECTED_SUBTYPE_LABELS_FULL = SELECTED_SUBTYPE_LABELS_REDUCED + ["DC", "DC2", "pDC"]

SELECTED_SUBTYPE_DISPLAY_MAP = {
    "Tem/Trm cytotoxic T cells": "Tem/Trm\ncytotoxic T",
    "Tem/Temra cytotoxic T cells": "Tem/Temra\ncytotoxic T",
    "Tcm/Naive cytotoxic T cells": "Tcm/Naive\ncytotoxic T",
    "Trm cytotoxic T cells": "Trm\ncytotoxic T",
    "Tcm/Naive helper T cells": "Tcm/Naive\nhelper T",
    "Tem/Effector helper T cells": "Tem/Effector\nhelper T",
    "Regulatory T cells": "Regulatory\nT cells",
    "CD16- NK cells": "CD16-\nNK",
    "CD16+ NK cells": "CD16+\nNK",
    "Naive B cells": "Naive\nB cells",
    "Memory B cells": "Memory\nB cells",
    "Plasma cells": "Plasma\ncells",
    "Classical monocytes": "Classical\nmonocytes",
    "Non-classical monocytes": "Non-classical\nmonocytes",
    "Macrophages": "Macrophages",
    "DC": "DC",
    "DC2": "DC2",
    "pDC": "pDC",
}


parser = ArgumentParser()
parser.add_argument("--input-h5ad", type=Path, required=True)
parser.add_argument("--output-h5ad", type=Path, required=True)
parser.add_argument("--fig-dir", type=Path, required=True)
parser.add_argument("--batch-key", default="batch")
parser.add_argument("--n-top-genes", type=int, default=2000)
parser.add_argument("--n-pcs", type=int, default=30)
parser.add_argument("--n-neighbors", type=int, default=15)
parser.add_argument("--leiden-resolution", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

args.fig_dir.mkdir(parents=True, exist_ok=True)
args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)

if args.output_h5ad.exists():
    raise FileExistsError(f"Output already exists: {args.output_h5ad}")

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
sc.settings.figdir = str(args.fig_dir)
sc.settings.verbosity = 3
sc.settings.set_figure_params(
    dpi=80,
    color_map="viridis",
    transparent=False,
    vector_friendly=True,
    frameon=False,
)

print(f"Input: {args.input_h5ad}")
print(f"Output: {args.output_h5ad}")
print(f"Seed: {args.seed}")

np.random.seed(args.seed)

adata = sc.read_h5ad(args.input_h5ad)
print(adata)

if "counts" not in adata.layers:
    raise ValueError("counts layer not found in input h5ad.")

if args.batch_key not in adata.obs.columns:
    raise ValueError(f"{args.batch_key} not found in adata.obs.")

adata.X = adata.layers["counts"]
if sp.issparse(adata.X):
    adata.X = adata.X.tocsr().astype(np.float32)
else:
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)

adata.raw = None

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=args.n_top_genes,
    batch_key=args.batch_key,
    subset=False,
)

n_hvg = int(adata.var["highly_variable"].sum())
print(f"Number of HVGs: {n_hvg}")

adata = adata[:, adata.var["highly_variable"]].copy()
print(f"Subsetted adata: {adata.shape}")

gc.collect()

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=args.n_pcs, svd_solver="arpack")

sce.pp.harmony_integrate(
    adata,
    key=args.batch_key,
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
)

sc.pp.neighbors(
    adata,
    n_neighbors=args.n_neighbors,
    n_pcs=args.n_pcs,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata, random_state=args.seed)

leiden_key = f"leiden_r{str(args.leiden_resolution).replace('.', '')}"
sc.tl.leiden(adata, resolution=args.leiden_resolution, key_added=leiden_key, random_state=args.seed)

print(adata.obs[leiden_key].value_counts().head())

sc.pl.umap(
    adata,
    color=leiden_key,
    legend_loc="on data",
    show=False,
    save=f"_umap_harmony_hvg{args.n_top_genes}_pc{args.n_pcs}_k{args.n_neighbors}_{leiden_key}.pdf",
)

umap_color_cols = [x for x in [args.batch_key, "Tissue", "sample"] if x in adata.obs.columns]
if len(umap_color_cols) > 0:
    sc.pl.umap(
        adata,
        color=umap_color_cols,
        ncols=min(3, len(umap_color_cols)),
        frameon=False,
        show=False,
        save=f"_umap_harmony_context_hvg{args.n_top_genes}_pc{args.n_pcs}_k{args.n_neighbors}.pdf",
    )

if "Tissue" in adata.obs.columns:
    tab = pd.crosstab(
        adata.obs["Tissue"].astype("object").fillna("NA"),
        adata.obs[leiden_key].astype("object").fillna("NA"),
    )
    tab_pct = tab.div(tab.sum(axis=1), axis=0).fillna(0).mul(100)
    tab_pct.to_csv(
        args.fig_dir / f"leiden_by_tissue_pct_hvg{args.n_top_genes}_pc{args.n_pcs}_k{args.n_neighbors}_{leiden_key}.tsv",
        sep="\t",
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    tab_pct.T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.8,
        edgecolor="none",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Leiden composition by tissue")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="Tissue")
    ax.grid(False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        args.fig_dir / f"leiden_by_tissue_stacked_pct_hvg{args.n_top_genes}_pc{args.n_pcs}_k{args.n_neighbors}_{leiden_key}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    sns.heatmap(
        tab_pct,
        ax=ax,
        cmap="viridis",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Percentage (%)", "shrink": 0.8},
    )
    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel("Tissue")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(
        args.fig_dir / f"leiden_by_tissue_heatmap_hvg{args.n_top_genes}_pc{args.n_pcs}_k{args.n_neighbors}_{leiden_key}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    selected_tissue_order = ["aqueous", "pbmc", "csf"]
    fig_specs = [
        ("reduced", SELECTED_SUBTYPE_LABELS_REDUCED, 4.8, "selected_subtypes_grouped_bar_reduced.pdf", "selected_subtypes_heatmap_x_tissue_y_subtype_reduced.pdf"),
        ("full", SELECTED_SUBTYPE_LABELS_FULL, 5.3, "selected_subtypes_grouped_bar_full.pdf", "selected_subtypes_heatmap_x_tissue_y_subtype_full.pdf"),
    ]
    keep_cols = [x for x in [args.batch_key, "Tissue", "sample", "predicted_labels_Immune_All_Low_leiden_r05", "majority_voting_Immune_All_Low_leiden_r05"] if x in adata.obs.columns]
    if "majority_voting_Immune_All_Low_leiden_r05" in adata.obs.columns:
        label_tab = pd.crosstab(
            adata.obs["Tissue"].astype("object").fillna("NA"),
            adata.obs["majority_voting_Immune_All_Low_leiden_r05"].astype("object").fillna("NA"),
        )
        for _, subtype_labels, fig_w, bar_name, heat_name in fig_specs:
            heatmap_keep = [x for x in subtype_labels if x in label_tab.columns]
            if len(heatmap_keep) == 0:
                continue
            plot_data = (
                label_tab.loc[[t for t in selected_tissue_order if t in label_tab.index], heatmap_keep]
                .div(label_tab.sum(axis=1), axis=0)
                .fillna(0)
                .mul(100)
            )
            plot_data.columns = [SELECTED_SUBTYPE_DISPLAY_MAP.get(x, x) for x in plot_data.columns]
            plot_data = plot_data[[SELECTED_SUBTYPE_DISPLAY_MAP.get(x, x) for x in heatmap_keep]]
            plot_data_t = plot_data.T
            plot_data_t.to_csv(args.fig_dir / heat_name.replace(".pdf", ".tsv"), sep="\t")

            fig, ax = plt.subplots(figsize=(fig_w, 3.8))
            x = np.arange(plot_data_t.shape[0])
            n_tissues = plot_data_t.shape[1]
            width = 0.8 / n_tissues
            offsets = np.linspace(-(n_tissues - 1) / 2, (n_tissues - 1) / 2, n_tissues) * width
            tissue_colors = {"aqueous": "#1F77B4", "pbmc": "#FF7F0E", "csf": "#2CA02C"}
            for i, tissue in enumerate(plot_data_t.columns):
                ax.bar(
                    x + offsets[i],
                    plot_data_t[tissue].values,
                    width=width,
                    color=tissue_colors.get(tissue, "#7F7F7F"),
                    label=tissue,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(plot_data_t.index, rotation=90, ha="center", fontsize=8)
            ax.set_ylabel("Percentage (%)")
            ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(False)
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(args.fig_dir / bar_name, dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(fig_w - 1.6, 5.2))
            im = ax.imshow(
                plot_data_t.values,
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=np.ceil(plot_data_t.to_numpy().max()),
            )
            ax.set_xticks(np.arange(plot_data_t.shape[1]))
            ax.set_xticklabels(plot_data_t.columns, fontsize=9, color="black")
            ax.set_yticks(np.arange(plot_data_t.shape[0]))
            ax.set_yticklabels(plot_data_t.index, fontsize=7, color="black")
            for i in range(plot_data_t.shape[0]):
                for j in range(plot_data_t.shape[1]):
                    val = plot_data_t.iloc[i, j]
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5, color="white" if val < 18 else "black")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis="x", colors="black", length=0, pad=2)
            ax.tick_params(axis="y", colors="black", length=0)
            cbar = fig.colorbar(im, ax=ax, shrink=0.82, fraction=0.05, pad=0.02)
            cbar.set_label("Percentage of all cells (%)", fontsize=9, color="black")
            cbar.ax.tick_params(labelsize=8, colors="black")
            fig.tight_layout()
            fig.savefig(args.fig_dir / heat_name, dpi=300, bbox_inches="tight")
            plt.close(fig)

adata.write_h5ad(args.output_h5ad, compression="gzip")
print(f"Saved: {args.output_h5ad}")
