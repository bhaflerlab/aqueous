#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import celltypist
from celltypist import models
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


broad_label_map = {
    "Age-associated B cells": "B / plasma",
    "Alveolar macrophages": "Myeloid / macrophage",
    "B cells": "B / plasma",
    "CD16+ NK cells": "NK / innate lymphoid",
    "CD16- NK cells": "NK / innate lymphoid",
    "CD8a/a": "CD8 T / cytotoxic T",
    "CD8a/b(entry)": "CD8 T / cytotoxic T",
    "CRTAM+ gamma-delta T cells": "Other lymphoid",
    "Classical monocytes": "Myeloid / macrophage",
    "Cycling B cells": "Cycling",
    "Cycling DCs": "Cycling",
    "Cycling NK cells": "Cycling",
    "Cycling T cells": "Cycling",
    "DC": "DC / pDC",
    "DC precursor": "Rare progenitor / precursor",
    "DC1": "DC / pDC",
    "DC2": "DC / pDC",
    "DC3": "DC / pDC",
    "Double-negative thymocytes": "Rare progenitor / precursor",
    "Double-positive thymocytes": "Rare progenitor / precursor",
    "ETP": "Rare progenitor / precursor",
    "Early erythroid": "Rare progenitor / precursor",
    "Early lymphoid/T lymphoid": "Rare progenitor / precursor",
    "Epithelial cells": "Non-immune / other",
    "Erythrocytes": "Non-immune / other",
    "Erythrophagocytic macrophages": "Myeloid / macrophage",
    "Fibroblasts": "Non-immune / other",
    "Follicular B cells": "B / plasma",
    "Follicular helper T cells": "Helper / regulatory T",
    "HSC/MPP": "Rare progenitor / precursor",
    "Hofbauer cells": "Myeloid / macrophage",
    "ILC2": "NK / innate lymphoid",
    "ILC3": "NK / innate lymphoid",
    "Intermediate macrophages": "Myeloid / macrophage",
    "Intestinal macrophages": "Myeloid / macrophage",
    "Kidney-resident macrophages": "Myeloid / macrophage",
    "MAIT cells": "Other lymphoid",
    "MEMP": "Rare progenitor / precursor",
    "Macrophages": "Myeloid / macrophage",
    "Mast cells": "Other lymphoid",
    "Megakaryocyte precursor": "Rare progenitor / precursor",
    "Memory B cells": "B / plasma",
    "Memory CD4+ cytotoxic T cells": "CD8 T / cytotoxic T",
    "Migratory DCs": "DC / pDC",
    "Mono-mac": "Myeloid / macrophage",
    "Monocyte precursor": "Rare progenitor / precursor",
    "Monocytes": "Myeloid / macrophage",
    "Myelocytes": "Rare progenitor / precursor",
    "NK cells": "NK / innate lymphoid",
    "NKT cells": "Other lymphoid",
    "Naive B cells": "B / plasma",
    "Neutrophil-myeloid progenitor": "Rare progenitor / precursor",
    "Non-classical monocytes": "Myeloid / macrophage",
    "Plasma cells": "B / plasma",
    "Plasmablasts": "B / plasma",
    "Pro-B cells": "Rare progenitor / precursor",
    "Proliferative germinal center B cells": "B / plasma",
    "Regulatory T cells": "Helper / regulatory T",
    "Tcm/Naive cytotoxic T cells": "CD8 T / cytotoxic T",
    "Tcm/Naive helper T cells": "Helper / regulatory T",
    "Tem/Effector helper T cells": "Helper / regulatory T",
    "Tem/Effector helper T cells PD1+": "Helper / regulatory T",
    "Tem/Temra cytotoxic T cells": "CD8 T / cytotoxic T",
    "Tem/Trm cytotoxic T cells": "CD8 T / cytotoxic T",
    "Transitional B cells": "B / plasma",
    "Transitional NK": "NK / innate lymphoid",
    "Treg(diff)": "Helper / regulatory T",
    "Trm cytotoxic T cells": "CD8 T / cytotoxic T",
    "Type 1 helper T cells": "Helper / regulatory T",
    "Type 17 helper T cells": "Helper / regulatory T",
    "gamma-delta T cells": "Other lymphoid",
    "pDC": "DC / pDC",
}

figure_broad_map = {
    "CD8 T / cytotoxic T": "CD8 / cytotoxic T",
    "Helper / regulatory T": "Helper / regulatory T",
    "NK / innate lymphoid": "NK / ILC",
    "B / plasma": "B / plasma",
    "Myeloid / macrophage": "Myeloid / macrophage",
    "DC / pDC": "DC / pDC",
    "Other lymphoid": "Other",
    "Cycling": "Other",
    "Rare progenitor / precursor": "Other",
    "Non-immune / other": "Other",
}

figure_broad_order = [
    "CD8 / cytotoxic T",
    "Helper / regulatory T",
    "NK / ILC",
    "B / plasma",
    "Myeloid / macrophage",
    "DC / pDC",
    "Other",
]

figure_broad_display_map = {
    "CD8 / cytotoxic T": "CD8 /\ncytotoxic T",
    "Helper / regulatory T": "Helper /\nregulatory T",
    "NK / ILC": "NK / ILC",
    "B / plasma": "B / plasma",
    "Myeloid / macrophage": "Myeloid /\nmacrophage",
    "DC / pDC": "DC / pDC",
    "Other": "Other",
}

tissue_order = ["aqueous", "pbmc", "csf"]
tissue_colors = {"aqueous": "#2ecc71", "pbmc": "#e74c3c", "csf": "#3498db"}

heatmap_display_map = {
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

heatmap_labels_reduced = [
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

heatmap_labels_full = heatmap_labels_reduced + ["DC", "DC2", "pDC"]

focus_labels = [
    "Tem/Trm cytotoxic T cells",
    "Tem/Temra cytotoxic T cells",
    "Tcm/Naive cytotoxic T cells",
    "Trm cytotoxic T cells",
    "Classical monocytes",
    "Non-classical monocytes",
    "Macrophages",
]

focus_display_map = {
    "Tem/Trm cytotoxic T cells": "Tem/Trm\ncytotoxic T",
    "Tem/Temra cytotoxic T cells": "Tem/Temra\ncytotoxic T",
    "Tcm/Naive cytotoxic T cells": "Tcm/Naive\ncytotoxic T",
    "Trm cytotoxic T cells": "Trm\ncytotoxic T",
    "Classical monocytes": "Classical\nmonocytes",
    "Non-classical monocytes": "Non-classical\nmonocytes",
    "Macrophages": "Macrophages",
}


def plot_grouped_pct_bar(tab, row_order, display_map, out_fp, figsize):
    tab = tab.loc[[t for t in tissue_order if t in tab.index], :]
    plot_data = tab.div(tab.sum(axis=1), axis=0).fillna(0).mul(100).T
    keep_rows = [x for x in row_order if x in plot_data.index]
    plot_data = plot_data.loc[keep_rows, [t for t in tissue_order if t in plot_data.columns]].copy()

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(plot_data.index))
    n_tissues = len(plot_data.columns)
    width = 0.8 / n_tissues
    offsets = np.linspace(-(n_tissues - 1) / 2, (n_tissues - 1) / 2, n_tissues) * width

    for i, tissue in enumerate(plot_data.columns):
        ax.bar(x + offsets[i], plot_data[tissue].values, width=width, color=tissue_colors[tissue], label=tissue)

    ax.set_xticks(x)
    ax.set_xticklabels([display_map.get(x, x) for x in plot_data.index], rotation=90, ha="center", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_x_tissue_y_subtype(tab_all, labels, out_fp, fig_w):
    keep = [x for x in labels if x in tab_all.columns]
    plot_data = tab_all.loc[[t for t in tissue_order if t in tab_all.index], keep].div(tab_all.sum(axis=1), axis=0).mul(100)
    plot_data.columns = [heatmap_display_map.get(x, x) for x in plot_data.columns]
    plot_data = plot_data[[heatmap_display_map.get(x, x) for x in keep]].T

    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    im = ax.imshow(plot_data.values, aspect="auto", cmap="viridis", vmin=0, vmax=np.ceil(plot_data.to_numpy().max()))
    ax.set_xticks(np.arange(plot_data.shape[1]))
    ax.set_xticklabels(plot_data.columns, fontsize=9, color="black")
    ax.set_yticks(np.arange(plot_data.shape[0]))
    ax.set_yticklabels(plot_data.index, fontsize=7, color="black")
    ax.grid(False)

    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            val = plot_data.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5, color="white" if val < 18 else "black")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="black", length=0, pad=2)
    ax.tick_params(axis="y", colors="black", length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, fraction=0.05, pad=0.02)
    cbar.set_label("Percentage of all cells (%)", fontsize=9, color="black")
    cbar.ax.tick_params(labelsize=8, colors="black")
    plt.tight_layout()
    fig.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_focus_heatmap(tab_focus, out_fp):
    plot_data = tab_focus.div(tab_focus.sum(axis=1), axis=0).mul(100).T
    plot_data.index = [focus_display_map.get(x, x) for x in plot_data.index]

    fig, ax = plt.subplots(figsize=(2.4, 3.7))
    im = ax.imshow(plot_data.values, aspect="auto", cmap="viridis", vmin=0, vmax=np.ceil(plot_data.to_numpy().max()))
    ax.set_xticks(np.arange(plot_data.shape[1]))
    ax.set_xticklabels(plot_data.columns, fontsize=9, color="black")
    ax.set_yticks(np.arange(plot_data.shape[0]))
    ax.set_yticklabels(plot_data.index, fontsize=7, color="black")
    ax.grid(False)

    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            val = plot_data.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5, color="white" if val < 18 else "black")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="black", length=0, pad=2)
    ax.tick_params(axis="y", colors="black", length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, fraction=0.08, pad=0.02)
    cbar.set_label("Percentage of tissue (%)", fontsize=9, color="black")
    cbar.ax.tick_params(labelsize=8, colors="black")
    plt.tight_layout()
    fig.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close(fig)


parser = ArgumentParser()
parser.add_argument("--full-h5ad", type=Path, required=True)
parser.add_argument("--harmony-h5ad", type=Path, required=True)
parser.add_argument("--model-path", type=Path, required=True)
parser.add_argument("--output-h5ad", type=Path, required=True)
parser.add_argument("--fig-dir", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--plot-source-h5ad", type=Path, default=None)
parser.add_argument("--leiden-col", default="leiden_r05")
parser.add_argument("--model-name", default="Immune_All_Low")
parser.add_argument("--tissue-col", default="Tissue")
args = parser.parse_args()

args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
args.fig_dir.mkdir(parents=True, exist_ok=True)
args.run_dir.mkdir(parents=True, exist_ok=True)
if args.output_h5ad.exists():
    raise FileExistsError(f"Output already exists: {args.output_h5ad}")

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
sc.settings.figdir = str(args.fig_dir)

adata_ct = sc.read(args.full_h5ad)
adata_hvg = sc.read(args.harmony_h5ad)

print("Loaded full-gene adata_ct:", adata_ct)
print("Loaded HVG adata_hvg:", adata_hvg)

if args.leiden_col not in adata_hvg.obs.columns:
    raise ValueError(f"{args.leiden_col} not found in HVG object.")

adata_ct.obs[args.leiden_col] = adata_hvg.obs[args.leiden_col].reindex(adata_ct.obs_names)
if adata_ct.obs[args.leiden_col].isna().any():
    raise ValueError(f"{args.leiden_col} contains NA after reindexing.")

if "counts" not in adata_ct.layers:
    raise ValueError("counts layer not found in adata_ct.")

adata_ct.X = adata_ct.layers["counts"].copy()
sc.pp.normalize_total(adata_ct, target_sum=1e4)
sc.pp.log1p(adata_ct)

print("model_path =", args.model_path)
_ = models.Model.load(model=str(args.model_path)).cell_types
print("Loaded model OK:", args.model_path)

pred = celltypist.annotate(
    adata_ct,
    model=str(args.model_path),
    majority_voting=True,
    over_clustering=args.leiden_col,
)

pl = pred.predicted_labels
col_pred = f"predicted_labels_{args.model_name}_{args.leiden_col}"
col_oc = f"over_clustering_{args.model_name}_{args.leiden_col}"
col_mv = f"majority_voting_{args.model_name}_{args.leiden_col}"

adata_ct.obs[col_pred] = pl["predicted_labels"].reindex(adata_ct.obs_names).astype("object")
adata_ct.obs[col_oc] = pl["over_clustering"].reindex(adata_ct.obs_names).astype("object")
adata_ct.obs[col_mv] = pl["majority_voting"].reindex(adata_ct.obs_names).astype("object")

if "X_umap" in adata_hvg.obsm:
    adata_ct.obsm["X_umap"] = adata_hvg.obsm["X_umap"].copy()

changed = adata_ct.obs[col_pred].astype(str) != adata_ct.obs[col_mv].astype(str)
print("Added columns:", [col_pred, col_oc, col_mv])
print("changed rate:", float(changed.mean()))

plot_adata = sc.read_h5ad(args.plot_source_h5ad) if args.plot_source_h5ad is not None else adata_ct
if col_pred not in plot_adata.obs.columns:
    if set(plot_adata.obs_names) != set(adata_ct.obs_names):
        raise ValueError(f"{col_pred} not found in plot source and obs_names do not match annotation output.")
    plot_adata.obs[col_pred] = adata_ct.obs[col_pred].reindex(plot_adata.obs_names)

obs_df = plot_adata.obs[[args.tissue_col, col_pred]].copy()
obs_df[args.tissue_col] = obs_df[args.tissue_col].astype("object").fillna("NA")
obs_df[col_pred] = obs_df[col_pred].astype("object").fillna("NA")
obs_df["broad_label"] = obs_df[col_pred].map(broad_label_map)

if obs_df["broad_label"].isna().any():
    unmapped = sorted(obs_df.loc[obs_df["broad_label"].isna(), col_pred].astype(str).unique().tolist())
    raise ValueError(f"Unmapped labels found: {unmapped}")

tab_broad = pd.crosstab(obs_df[args.tissue_col], obs_df["broad_label"])
tab_broad.to_csv(args.run_dir / "HCplusAQ.labels_by_Tissue_pred_broad.tsv", sep="\t")

obs_df_fig = obs_df.copy()
obs_df_fig["figure_broad_label"] = obs_df_fig["broad_label"].map(figure_broad_map)
if obs_df_fig["figure_broad_label"].isna().any():
    missing = sorted(obs_df_fig.loc[obs_df_fig["figure_broad_label"].isna(), "broad_label"].astype(str).unique().tolist())
    raise ValueError(f"Unmapped figure broad labels found: {missing}")

tab_broad_fig = pd.crosstab(obs_df_fig[args.tissue_col], obs_df_fig["figure_broad_label"])
tab_broad_fig.to_csv(args.run_dir / "HCplusAQ.labels_by_Tissue_pred_broad_figure.tsv", sep="\t")

plot_grouped_pct_bar(
    tab_broad_fig,
    figure_broad_order,
    figure_broad_display_map,
    args.fig_dir / "HCplusAQ_broad_immune_grouped_bar_v01.pdf",
    (8.5, 4.0),
)

tab_all = pd.crosstab(
    plot_adata.obs[args.tissue_col].astype("object").fillna("NA"),
    plot_adata.obs[col_pred].astype("object").fillna("NA"),
)
plot_heatmap_x_tissue_y_subtype(
    tab_all,
    heatmap_labels_reduced,
    args.fig_dir / "HCplusAQ_selected_subtypes_heatmap_x_tissue_y_subtype_v01.pdf",
    2.6,
)
plot_heatmap_x_tissue_y_subtype(
    tab_all,
    heatmap_labels_full,
    args.fig_dir / "HCplusAQ_selected_subtypes_heatmap_x_tissue_y_subtype_full_v01.pdf",
    3.0,
)

tab_focus = pd.crosstab(
    plot_adata.obs[args.tissue_col].astype("object").fillna("NA"),
    plot_adata.obs[col_pred].astype("object").fillna("NA"),
)
focus_keep = [x for x in focus_labels if x in tab_focus.columns]
tab_focus = tab_focus.loc[[t for t in tissue_order if t in tab_focus.index], focus_keep].copy()
tab_focus.to_csv(args.run_dir / "HCplusAQ.labels_by_Tissue_pred_cd8_myeloid_focus.tsv", sep="\t")

plot_grouped_pct_bar(
    tab_focus,
    focus_labels,
    focus_display_map,
    args.fig_dir / "HCplusAQ_cd8_myeloid_grouped_bar_v01.pdf",
    (4.3, 4.0),
)
plot_focus_heatmap(tab_focus, args.fig_dir / "HCplusAQ_cd8_myeloid_heatmap_v01.pdf")

adata_ct.write_h5ad(args.output_h5ad, compression="gzip")
print(f"Saved: {args.output_h5ad}")
