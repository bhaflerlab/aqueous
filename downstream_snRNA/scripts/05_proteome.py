#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


LINEAGE_MAP = {
    "Tem/Trm cytotoxic T cells": "CD8_cytotoxic_T",
    "Tem/Temra cytotoxic T cells": "CD8_cytotoxic_T",
    "Tcm/Naive cytotoxic T cells": "CD8_cytotoxic_T",
    "Trm cytotoxic T cells": "CD8_cytotoxic_T",
    "CRTAM+ gamma-delta T cells": "CD8_cytotoxic_T",
    "MAIT cells": "CD8_cytotoxic_T",
    "Type 1 helper T cells": "Helper_T",
    "Tem/Effector helper T cells": "Helper_T",
    "Tcm/Naive helper T cells": "Helper_T",
    "Follicular helper T cells": "Helper_T",
    "Type 17 helper T cells": "Helper_T",
    "Regulatory T cells": "Regulatory_T",
    "CD16- NK cells": "NK",
    "CD16+ NK cells": "NK",
    "NK cells": "NK",
    "ILC3": "NK",
    "Memory B cells": "B_plasma",
    "Naive B cells": "B_plasma",
    "Plasma cells": "B_plasma",
    "Age-associated B cells": "B_plasma",
    "Proliferative germinal center B cells": "B_plasma",
    "Classical monocytes": "Monocyte_macrophage",
    "Non-classical monocytes": "Monocyte_macrophage",
    "Macrophages": "Monocyte_macrophage",
    "Erythrophagocytic macrophages": "Monocyte_macrophage",
    "Alveolar macrophages": "Monocyte_macrophage",
    "Intestinal macrophages": "Monocyte_macrophage",
    "Intermediate macrophages": "Monocyte_macrophage",
    "Mono-mac": "Monocyte_macrophage",
    "Hofbauer cells": "Monocyte_macrophage",
    "DC": "DC",
    "DC1": "DC",
    "DC2": "DC",
    "DC precursor": "DC",
    "Fibroblasts": "Other_stromal",
}

BROAD_ORDER = [
    "CD8_cytotoxic_T",
    "Helper_T",
    "Regulatory_T",
    "NK",
    "B_plasma",
    "Monocyte_macrophage",
    "DC",
    "Other_stromal",
]

REACTOME_TERMS_KEEP = [
    "REACTOME_COMPLEMENT_CASCADE",
    "REACTOME_INNATE_IMMUNE_SYSTEM",
    "REACTOME_HEMOSTASIS",
    "REACTOME_INITIAL_TRIGGERING_OF_COMPLEMENT",
    "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION",
]

GOBP_TERMS_KEEP = [
    "GOBP_COMPLEMENT_ACTIVATION",
    "GOBP_HUMORAL_IMMUNE_RESPONSE",
    "GOBP_IMMUNE_EFFECTOR_PROCESS",
    "GOBP_B_CELL_MEDIATED_IMMUNITY",
    "GOBP_INFLAMMATORY_RESPONSE",
]

MODULE_ORDER = [
    "complement_innate",
    "protease_hemostasis",
    "humoral_immunoglobulin",
    "ecm_adhesion",
    "immune_regulation",
]

CELLTYPE_LABEL_MAP = {
    "CD8_cytotoxic_T": "CD8 T",
    "Helper_T": "CD4 T",
    "Regulatory_T": "Treg",
    "NK": "NK",
    "B_plasma": "B/Plasma",
    "Monocyte_macrophage": "Mono/Mac",
    "DC": "DC",
    "Other_stromal": "Stromal/Other",
}

MODULE_LABEL_MAP = {
    "complement_innate": "Complement/innate",
    "protease_hemostasis": "Protease/Hemostasis",
    "humoral_immunoglobulin": "Humoral/Ig",
    "ecm_adhesion": "ECM/Adhesion",
    "immune_regulation": "Immune/Regulation",
}


def build_term_gene_table(enrich_df, term_list, source_name):
    rows = []
    for term in term_list:
        sub = enrich_df.loc[enrich_df["Term"] == term].copy()
        if sub.shape[0] == 0:
            continue
        genes_str = sub.iloc[0]["Genes"]
        if pd.isna(genes_str):
            continue
        genes = [g.strip() for g in str(genes_str).split(";") if g.strip()]
        for gene in genes:
            rows.append({"gene": gene, "term": term, "source": source_name})
    return pd.DataFrame(rows)


parser = ArgumentParser()
parser.add_argument("--proteome-csv", type=Path, required=True)
parser.add_argument("--input-h5ad", type=Path, required=True)
parser.add_argument("--go-bp-gmt", type=Path, required=True)
parser.add_argument("--reactome-gmt", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--fig-dir", type=Path, required=True)
parser.add_argument("--cell-type-col", default="predicted_labels_Immune_All_Low_leiden_r05")
parser.add_argument("--broad-col", default="broad_lineage_v01")
parser.add_argument("--manual-broad-col", default="broad_lineage_v01")
parser.add_argument("--expr-layer", default="X", choices=["X", "counts"])
parser.add_argument("--broad-mode", default="both", choices=["celltypist", "manual", "both"])
parser.add_argument("--detect-threshold", type=float, default=0.5)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
args.fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.proteome_csv)
adata = sc.read_h5ad(args.input_h5ad)

if args.expr_layer == "counts":
    if "counts" not in adata.layers:
        raise ValueError("counts layer not found in input h5ad.")
    adata.X = adata.layers["counts"].copy()
else:
    if "log1p" not in adata.uns:
        print("Warning: using adata.X without explicit log1p metadata.")

df["Detected_prop_num"] = (
    df["Detected in Proportion of Samples (%)"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
    .astype(float)
)
df["Gene Symbol"] = df["Gene Symbol"].astype(str).str.strip()

df_detected = df.loc[df["Detected_prop_num"] >= args.detect_threshold].copy()
adata_genes = pd.Index(adata.var_names.astype(str)).str.strip()
proteome_genes = pd.Index(df_detected["Gene Symbol"].astype(str)).str.strip()
shared_genes = proteome_genes.intersection(adata_genes)

df_50_in_h5ad = df_detected.loc[df_detected["Gene Symbol"].isin(shared_genes)].copy()
df_50_in_h5ad = df_50_in_h5ad.sort_values(
    by=["Detected_prop_num", "Mean PSM value"],
    ascending=[False, False],
).reset_index(drop=True)
df_50_in_h5ad.to_csv(args.output_dir / "aq_proteome_genes_detected50_in_h5ad_v01.csv", index=False)

with open(args.output_dir / "aq_proteome_genes_detected50_in_h5ad_gene_list_v01.txt", "w") as fout:
    for gene in df_50_in_h5ad["Gene Symbol"].dropna().astype(str).drop_duplicates().sort_values().tolist():
        fout.write(f"{gene}\n")

adata_aq = adata[adata.obs["Tissue"] == "aqueous"].copy()
proteome_genes_aq = [g for g in proteome_genes if g in adata_aq.var_names]

celltypist_broad_col = f"{args.broad_col}_celltypist"
adata_aq.obs[celltypist_broad_col] = (
    adata_aq.obs[args.cell_type_col]
    .astype(str)
    .map(LINEAGE_MAP)
    .fillna("Other")
)

available_broad_specs = []
if args.broad_mode in {"celltypist", "both"}:
    available_broad_specs.append(("celltypist", celltypist_broad_col, "aq_proteome185_broad"))
if args.broad_mode in {"manual", "both"}:
    if args.manual_broad_col not in adata_aq.obs.columns:
        raise ValueError(f"manual broad column not found: {args.manual_broad_col}")
    available_broad_specs.append(("manual", args.manual_broad_col, "aq_proteome185_manual_broad"))

for broad_mode_name, broad_col_name, prefix in available_broad_specs:
    mean_expr_broad = pd.DataFrame(index=proteome_genes_aq)
    for ct in adata_aq.obs[broad_col_name].astype(str).unique():
        if ct == "Other":
            continue
        mask = adata_aq.obs[broad_col_name].astype(str) == ct
        vals = adata_aq[mask, proteome_genes_aq].X.mean(axis=0)
        mean_expr_broad[ct] = np.asarray(vals).ravel()

    mean_expr_broad = mean_expr_broad[[c for c in BROAD_ORDER if c in mean_expr_broad.columns]].copy()
    mean_expr_broad_filt = mean_expr_broad.loc[mean_expr_broad.max(axis=1) >= 0.02].copy()
    row_means = mean_expr_broad_filt.mean(axis=1)
    row_stds = mean_expr_broad_filt.std(axis=1)
    mean_expr_broad_filt_z = (
        mean_expr_broad_filt
        .sub(row_means, axis=0)
        .div(row_stds.replace(0, np.nan), axis=0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    mean_expr_broad_filt.to_csv(args.output_dir / f"{prefix}_mean_expr_filt_v01.csv")
    mean_expr_broad_filt_z.to_csv(args.output_dir / f"{prefix}_mean_expr_filt_rowz_v01.csv")

    g = sns.clustermap(
        mean_expr_broad_filt_z,
        cmap="vlag",
        center=0,
        figsize=(4, 10),
        xticklabels=True,
        yticklabels=False,
        row_cluster=True,
        col_cluster=False,
        linewidths=0,
        linecolor="none",
    )
    g.ax_heatmap.grid(False)
    g.ax_heatmap.set_facecolor("white")
    g.savefig(args.fig_dir / f"{prefix}_rowz_clustermap_filt_v01.pdf", bbox_inches="tight")
    plt.close(g.fig)

proteome_gene_list = (
    df_50_in_h5ad["Gene Symbol"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)

go_bp_res = gp.enrich(
    gene_list=proteome_gene_list,
    gene_sets=str(args.go_bp_gmt),
    outdir=None,
    no_plot=True,
    background=list(adata.var_names),
)
go_bp_df = go_bp_res.results.copy().sort_values("Adjusted P-value").reset_index(drop=True)
go_bp_df["neglog10_fdr"] = -np.log10(go_bp_df["Adjusted P-value"].clip(lower=1e-300))
go_bp_df["overlap_n"] = go_bp_df["Overlap"].str.split("/").str[0].astype(int)
go_bp_df.to_csv(args.output_dir / "aq_proteome185_go_bp_ora_v01.csv", index=False)

reactome_res = gp.enrich(
    gene_list=proteome_gene_list,
    gene_sets=str(args.reactome_gmt),
    outdir=None,
    no_plot=True,
    background=list(adata.var_names),
)
reactome_df = reactome_res.results.copy().sort_values("Adjusted P-value").reset_index(drop=True)
reactome_df["neglog10_fdr"] = -np.log10(reactome_df["Adjusted P-value"].clip(lower=1e-300))
reactome_df["overlap_n"] = reactome_df["Overlap"].str.split("/").str[0].astype(int)
reactome_df.to_csv(args.output_dir / "aq_proteome185_reactome_ora_v01.csv", index=False)

for name, plot_df in [("reactome", reactome_df.head(15).iloc[::-1]), ("go_bp", go_bp_df.head(15).iloc[::-1])]:
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.barh(plot_df["Term"], plot_df["neglog10_fdr"])
    ax.set_xlabel("-log10(FDR)")
    ax.set_ylabel("")
    sns.despine()
    fig.savefig(args.fig_dir / f"aq_proteome185_{name}_ora_top15_v01.pdf", bbox_inches="tight")
    plt.close(fig)

annot_df = (
    df_50_in_h5ad[["Gene Symbol", "Detected_prop_num", "Mean PSM value"]]
    .copy()
    .rename(columns={"Gene Symbol": "gene", "Mean PSM value": "mean_psm"})
)
annot_df["gene"] = annot_df["gene"].astype(str).str.strip()
annot_df = annot_df.dropna(subset=["gene"]).drop_duplicates(subset=["gene"]).reset_index(drop=True)

term_gene_df = pd.concat(
    [
        build_term_gene_table(reactome_df, REACTOME_TERMS_KEEP, "reactome"),
        build_term_gene_table(go_bp_df, GOBP_TERMS_KEEP, "gobp"),
    ],
    axis=0,
    ignore_index=True,
).drop_duplicates().reset_index(drop=True)
term_gene_df.to_csv(args.output_dir / "aq_proteome185_term_gene_table_v01.csv", index=False)

gene_term_summary = (
    term_gene_df.groupby("gene")
    .agg(
        reactome_terms=("term", lambda x: ";".join(sorted(set([t for t in x if t.startswith("REACTOME_")])))),
        gobp_terms=("term", lambda x: ";".join(sorted(set([t for t in x if t.startswith("GOBP_")])))),
    )
    .reset_index()
)
annot_df = annot_df.merge(gene_term_summary, on="gene", how="left")
annot_df["reactome_terms"] = annot_df["reactome_terms"].fillna("")
annot_df["gobp_terms"] = annot_df["gobp_terms"].fillna("")

annot_df["is_complement"] = annot_df["gene"].str.startswith("C", na=False) & annot_df["gene"].isin(
    ["C1R", "C1S", "C2", "C3", "C4A", "C4B", "C5", "CFB", "CFH", "CFI", "SERPING1"]
)
annot_df["is_immunoglobulin"] = annot_df["gene"].str.startswith(("IGH", "IGK", "IGL"), na=False)
annot_df["is_hemostasis_like"] = annot_df["reactome_terms"].str.contains("HEMOSTASIS|PLATELET", regex=True, na=False)
annot_df["is_ecm_like"] = (
    annot_df["reactome_terms"].str.contains("EXTRACELLULAR_MATRIX", regex=True, na=False)
    | annot_df["gene"].isin(["COL9A3", "PCOLCE", "PAPLN", "VASN", "NRCAM", "LTBP2", "SPOCK1"])
)

annot_df["module_primary"] = "unassigned"
annot_df.loc[annot_df["is_immunoglobulin"], "module_primary"] = "humoral_immunoglobulin"
annot_df.loc[(annot_df["module_primary"] == "unassigned") & annot_df["is_complement"], "module_primary"] = "complement_innate"
annot_df.loc[(annot_df["module_primary"] == "unassigned") & annot_df["is_hemostasis_like"], "module_primary"] = "protease_hemostasis"
annot_df.loc[(annot_df["module_primary"] == "unassigned") & annot_df["is_ecm_like"], "module_primary"] = "ecm_adhesion"
annot_df.loc[
    (annot_df["module_primary"] == "unassigned")
    & annot_df["gobp_terms"].str.contains("INFLAMMATORY_RESPONSE|IMMUNE_EFFECTOR_PROCESS", regex=True, na=False),
    "module_primary",
] = "immune_regulation"
annot_df.loc[
    (annot_df["module_primary"] == "unassigned")
    & annot_df["reactome_terms"].str.contains("INSULIN_LIKE_GROWTH_FACTOR|SCAVENGING|HEME", regex=True, na=False),
    "module_primary",
] = "transport_scavenging"
annot_df.loc[annot_df["module_primary"] == "unassigned", "module_primary"] = "other_background_like"
annot_df.to_csv(args.output_dir / "aq_proteome185_gene_annotation_table_v01.csv", index=False)

module_genes_dict = {}
for module_name in MODULE_ORDER:
    genes = (
        annot_df.loc[annot_df["module_primary"] == module_name, "gene"]
        .dropna()
        .astype(str)
        .tolist()
    )
    module_genes_dict[module_name] = sorted(set([g for g in genes if g in adata_aq.var_names]))

adata_aq_score = adata_aq.copy()
score_modules = [m for m, genes in module_genes_dict.items() if len(genes) >= 3]
for module_name in score_modules:
    sc.tl.score_genes(
        adata_aq_score,
        gene_list=module_genes_dict[module_name],
        score_name=f"{module_name}_score",
        use_raw=False,
    )

module_gene_rows = []
for module_name, genes in module_genes_dict.items():
    for gene in genes:
        module_gene_rows.append({"module": module_name, "gene": gene})
pd.DataFrame(module_gene_rows).to_csv(args.output_dir / "aq_proteome185_module_gene_membership_v01.csv", index=False)

score_cols = [f"{m}_score" for m in score_modules]
for broad_mode_name, broad_col_name, prefix in available_broad_specs:
    score_summary = (
        adata_aq_score.obs[[broad_col_name] + score_cols]
        .groupby(broad_col_name, observed=True)
        .mean()
        .reset_index()
    )

    plot_df = score_summary.set_index(broad_col_name).copy()
    plot_df = plot_df.loc[[c for c in BROAD_ORDER if c in plot_df.index], :]
    plot_df.columns = [c.replace("_score", "") for c in plot_df.columns]
    plot_df_z = plot_df.T.copy()
    plot_df_z = plot_df_z.sub(plot_df_z.mean(axis=1), axis=0).div(plot_df_z.std(axis=1).replace(0, np.nan), axis=0).fillna(0).T

    module_n_map = {
        module_name: len(module_genes_dict[module_name])
        for module_name in plot_df_z.columns
        if module_name in module_genes_dict
    }
    plot_df_disp = plot_df_z.copy()
    plot_df_disp.index = [CELLTYPE_LABEL_MAP.get(x, x) for x in plot_df_disp.index]
    plot_df_disp.columns = [f"{MODULE_LABEL_MAP.get(x, x)}\n(n={module_n_map.get(x, 0)})" for x in plot_df_disp.columns]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sns.heatmap(
        plot_df_disp,
        cmap="vlag",
        center=0,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Module-wise z-score"},
        ax=ax,
    )
    ax.set_title(f"Proteome-derived modules ({broad_mode_name})", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=60)
    ax.tick_params(axis="y", rotation=0)
    sns.despine()
    fig.savefig(args.fig_dir / f"{prefix}_module_score_heatmap_rowz_n_v01.pdf", bbox_inches="tight")
    plt.close(fig)
