#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import textwrap

import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


ANALYSIS_LABELS = {
    "CD8_4labels": [
        "Tem/Trm cytotoxic T cells",
        "Tem/Temra cytotoxic T cells",
        "Tcm/Naive cytotoxic T cells",
        "Trm cytotoxic T cells",
    ],
    "Monocyte_like_3labels": [
        "Mono-mac",
        "Classical monocytes",
        "Non-classical monocytes",
    ],
    "Monocyte_2labels": [
        "Classical monocytes",
        "Non-classical monocytes",
    ],
}

GENE_PANEL_MAP = {
    "CD8_4labels": [
        "CD69",
        "CXCR4",
        "CD44",
        "TNFAIP3",
        "REL",
        "ZFP36",
        "CBLB",
        "SLC2A3",
        "SAT1",
        "ID2",
        "RGCC",
        "SIK3",
    ],
    "Monocyte_like_3labels": [
        "SLC2A3",
        "NAMPT",
        "ADM",
        "AQP9",
        "ACSL1",
        "FKBP5",
        "CHST11",
        "IFI30",
        "CD74",
        "CST3",
        "S100A8",
        "S100A9",
    ],
    "Monocyte_2labels": [
        "SLC2A3",
        "NAMPT",
        "ADM",
        "AQP9",
        "ACSL1",
        "FKBP5",
        "CHST11",
        "IFI30",
        "CD74",
        "CST3",
        "S100A8",
        "S100A9",
    ],
}

HALLMARK_SELECTED_TERMS = {
    "CD8_4labels": [
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_HYPOXIA",
        "HALLMARK_APOPTOSIS",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    ],
    "Monocyte_like_3labels": [
        "HALLMARK_HYPOXIA",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    ],
    "Monocyte_2labels": [
        "HALLMARK_HYPOXIA",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    ],
}

GROUP_LABEL_MAP = {
    "CD8_4labels": "CD8",
    "Monocyte_like_3labels": "Myeloid",
    "Monocyte_2labels": "Monocyte",
}

COMPARISON_ORDER = ["Aqueous_vs_PBMC", "Aqueous_vs_CSF"]
COMPARISON_TITLE_MAP = {
    "Aqueous_vs_PBMC": "Aqueous vs PBMC",
    "Aqueous_vs_CSF": "Aqueous vs CSF",
}
COLLECTIONS = ["hallmark", "reactome", "gobp"]
TISSUE_ORDER = ["aqueous", "pbmc", "csf"]


def clean_term_name(term):
    term = str(term)
    term = term.replace("HALLMARK_", "")
    term = term.replace("REACTOME_", "")
    term = term.replace("GOBP_", "")
    term = term.replace("GO_BP_", "")
    term = term.replace("_", " ").strip()
    return term


def read_deg_tsv(fp):
    df = pd.read_csv(fp, sep="\t").copy()
    for col in ["expr_diff_mean", "p_adj_fisher"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {fp}")
    df["expr_diff_mean"] = pd.to_numeric(df["expr_diff_mean"], errors="coerce")
    df["p_adj_fisher"] = pd.to_numeric(df["p_adj_fisher"], errors="coerce")
    df = df.dropna(subset=["gene", "expr_diff_mean", "p_adj_fisher"]).copy()
    return df


def make_prerank_df(deg_df):
    prerank_df = (
        deg_df[["gene", "expr_diff_mean"]]
        .dropna()
        .groupby("gene", as_index=False)["expr_diff_mean"]
        .mean()
        .sort_values("expr_diff_mean", ascending=False)
        .reset_index(drop=True)
    )
    return prerank_df


def run_prerank_and_save(prerank_df, gene_set_fp, out_fp, threads):
    pre_res = gp.prerank(
        rnk=prerank_df,
        gene_sets=str(gene_set_fp),
        threads=threads,
        min_size=10,
        max_size=500,
        permutation_num=1000,
        seed=123,
        outdir=None,
        ascending=False,
        verbose=False,
    )
    res2d = pre_res.res2d.copy()
    res2d.to_csv(out_fp, sep="\t", index=False)
    return res2d


def standardize_gsea_df(df, group_name, comparison, collection):
    rename_map = {
        "FDR q-val": "FDR",
        "NOM p-val": "NOM_pval",
        "FWER p-val": "FWER_pval",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    required = ["Term", "NES", "FDR"]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing GSEA columns {missing} for {group_name} {comparison} {collection}")
    df["group"] = group_name
    df["comparison"] = comparison
    df["collection"] = collection
    df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    if "NOM_pval" in df.columns:
        df["NOM_pval"] = pd.to_numeric(df["NOM_pval"], errors="coerce")
    df = df.dropna(subset=["Term", "NES", "FDR"]).copy()
    df["neglog10_FDR"] = -np.log10(df["FDR"].clip(lower=1e-300))
    return df


def save_hallmark_barplot(plot_df, group_name, fig_dir):
    selected_terms = HALLMARK_SELECTED_TERMS[group_name]
    term_display_order = selected_terms[::-1]
    fig, axes = plt.subplots(1, len(COMPARISON_ORDER), figsize=(7.8, 3.6), sharey=True)
    if len(COMPARISON_ORDER) == 1:
        axes = [axes]

    sub_all = plot_df[(plot_df["group"] == group_name) & (plot_df["Term"].isin(selected_terms))].copy()
    if sub_all.empty:
        raise ValueError(f"No hallmark rows found for {group_name}")

    x_lim = max(1.0, np.ceil(np.abs(sub_all["NES"]).max() * 10) / 10)
    for ax, comparison in zip(axes, COMPARISON_ORDER):
        sub = plot_df[
            (plot_df["group"] == group_name)
            & (plot_df["comparison"] == comparison)
            & (plot_df["Term"].isin(selected_terms))
        ].copy()
        sub["Term"] = pd.Categorical(sub["Term"], categories=term_display_order, ordered=True)
        sub = sub.sort_values("Term").reset_index(drop=True)

        y = np.arange(sub.shape[0])
        nes = sub["NES"].to_numpy()
        fdr = sub["FDR"].to_numpy()
        colors = ["#FF1A1A" if x > 0 else "#1A1AFF" for x in nes]

        ax.barh(y=y, width=nes, left=0, color=colors, edgecolor="none", height=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.fill(clean_term_name(x), width=18) for x in sub["Term"].astype(str)], fontsize=8)
        ax.set_title(COMPARISON_TITLE_MAP[comparison], fontsize=11)
        ax.set_xlabel("NES")
        ax.axvline(0, linestyle="--", linewidth=0.8, color="gray", alpha=0.8)
        ax.set_xlim(-x_lim, x_lim)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        text_offset = x_lim * 0.06
        for yi, xi, qi in zip(y, nes, fdr):
            if qi < 0.001:
                sig_text = "***"
            elif qi < 0.01:
                sig_text = "**"
            elif qi < 0.05:
                sig_text = "*"
            else:
                sig_text = ""
            if sig_text == "":
                continue
            ax.text(
                xi + text_offset,
                yi,
                sig_text,
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="white" if xi < 0 else "black",
            )

    axes[0].set_ylabel("Hallmark pathway")
    fig.suptitle(f"Hallmark GSEA: {GROUP_LABEL_MAP[group_name]}", fontsize=12, y=0.98)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color="#FF1A1A"), plt.Rectangle((0, 0), 1, 1, color="#1A1AFF")]
    fig.legend(
        legend_handles,
        ["Aqueous up (NES > 0)", "Aqueous down (NES < 0)"],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
    )
    plt.tight_layout()
    out_pdf = fig_dir / f"GSEA_hallmark_barplot_{group_name}_v01.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_hallmark_bubble(plot_df, group_names, fig_dir):
    selected_terms = sorted({x for g in group_names for x in HALLMARK_SELECTED_TERMS[g]})
    fig, axes = plt.subplots(1, len(COMPARISON_ORDER), figsize=(10, 5), sharey=True)
    if len(COMPARISON_ORDER) == 1:
        axes = [axes]

    x_map = {g: i for i, g in enumerate(group_names)}
    term_offset_map = {
        term: offset
        for term, offset in zip(sorted(selected_terms), np.linspace(-0.22, 0.22, len(selected_terms)))
    }
    group_color_map = {
        group_names[0]: "#4A90E2",
        group_names[1]: "#8E63CE" if len(group_names) > 1 else "#4A90E2",
    }

    for ax, comparison in zip(axes, COMPARISON_ORDER):
        sub = plot_df[(plot_df["comparison"] == comparison) & (plot_df["Term"].isin(selected_terms))].copy()
        sub["x"] = sub["group"].map(x_map)
        sub["x_jitter"] = sub["x"] + sub["Term"].map(term_offset_map)
        lab = sub[sub["FDR"] < 0.05].copy().sort_values("neglog10_FDR", ascending=False)

        for group_name in group_names:
            sg = sub[sub["group"] == group_name].copy()
            lg = lab[lab["group"] == group_name].copy()
            group_color = group_color_map[group_name]
            size_all = np.clip(np.abs(sg["NES"]) * 45, 20, 180)
            size_lab = np.clip(np.abs(lg["NES"]) * 45, 20, 180)
            ax.scatter(sg["x_jitter"], sg["neglog10_FDR"], s=size_all, color=group_color, alpha=1, edgecolor="none")
            ax.scatter(lg["x_jitter"], lg["neglog10_FDR"], s=size_lab, color=group_color, alpha=1, edgecolor="none")
            for _, row in lg.iterrows():
                ax.text(
                    row["x_jitter"] + 0.015,
                    row["neglog10_FDR"] + 0.5,
                    clean_term_name(row["Term"]),
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

        ax.set_title(COMPARISON_TITLE_MAP[comparison], fontsize=11)
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels([GROUP_LABEL_MAP[g] for g in group_names])
        ax.set_xlim(-0.45, len(group_names) - 0.55)
        ax.axhline(-np.log10(0.05), linestyle="--", linewidth=0.8, color="gray", alpha=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("-log10(FDR q-value)")
    fig.suptitle("Hallmark pathways enriched in aqueous-up programs", fontsize=13, y=0.98)
    size_values = [1.5, 2.0, 2.5, 3.0]
    size_handles = [plt.scatter([], [], s=np.clip(v * 45, 20, 180), color="gray", alpha=0.6) for v in size_values]
    fig.legend(
        size_handles,
        [f"NES = {v}" for v in size_values],
        title="Bubble size (NES)",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
    )
    plt.tight_layout()
    out_pdf = fig_dir / "Bubble_Hallmark_aqueous_up_sig_vs_pbmc_csf_v01.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_common_gene_panel(group_name, deg_files, adata, fig_dir, calc_dir, label_col, tissue_col, effect_cut, fdr_cut):
    deg_pbmc = read_deg_tsv(deg_files["Aqueous_vs_PBMC"])
    deg_csf = read_deg_tsv(deg_files["Aqueous_vs_CSF"])

    deg_pbmc_up = deg_pbmc.loc[(deg_pbmc["p_adj_fisher"] < fdr_cut) & (deg_pbmc["expr_diff_mean"] > effect_cut), ["gene", "expr_diff_mean", "p_adj_fisher"]].copy()
    deg_csf_up = deg_csf.loc[(deg_csf["p_adj_fisher"] < fdr_cut) & (deg_csf["expr_diff_mean"] > effect_cut), ["gene", "expr_diff_mean", "p_adj_fisher"]].copy()
    set_pbmc = set(deg_pbmc_up["gene"])
    set_csf = set(deg_csf_up["gene"])
    common_up = set_pbmc & set_csf

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    venn2(subsets=(set_pbmc, set_csf), set_labels=("Aqueous vs PBMC", "Aqueous vs CSF"), ax=ax)
    ax.set_title(f"Common Aqueous-up genes in {GROUP_LABEL_MAP[group_name]}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{GROUP_LABEL_MAP[group_name].lower()}_aq_up_venn_v01.pdf", bbox_inches="tight")
    plt.close(fig)

    common_pbmc = deg_pbmc_up.loc[deg_pbmc_up["gene"].isin(common_up)].copy()
    common_csf = deg_csf_up.loc[deg_csf_up["gene"].isin(common_up)].copy()
    common_tbl = common_pbmc.merge(common_csf, on="gene", suffixes=("_vs_pbmc", "_vs_csf"))
    common_tbl["mean_effect"] = (common_tbl["expr_diff_mean_vs_pbmc"] + common_tbl["expr_diff_mean_vs_csf"]) / 2
    common_tbl["min_effect"] = common_tbl[["expr_diff_mean_vs_pbmc", "expr_diff_mean_vs_csf"]].min(axis=1)
    common_tbl["max_fdr"] = common_tbl[["p_adj_fisher_vs_pbmc", "p_adj_fisher_vs_csf"]].max(axis=1)
    common_tbl = common_tbl.sort_values(["min_effect", "mean_effect", "max_fdr"], ascending=[False, False, True]).reset_index(drop=True)
    common_tbl.to_csv(calc_dir / f"{GROUP_LABEL_MAP[group_name].lower()}_aq_up_common_genes_v01.tsv", sep="\t", index=False)

    genes_use = [g for g in GENE_PANEL_MAP[group_name] if g in adata.var_names]
    label_values = ANALYSIS_LABELS[group_name]
    adata_sub = adata[adata.obs[label_col].isin(label_values), genes_use].copy()
    adata_sub.obs[tissue_col] = pd.Categorical(adata_sub.obs[tissue_col], categories=TISSUE_ORDER, ordered=True)

    dp = sc.pl.dotplot(
        adata_sub,
        var_names=genes_use,
        groupby=tissue_col,
        categories_order=TISSUE_ORDER,
        standard_scale="var",
        dendrogram=False,
        expression_cutoff=0.0,
        colorbar_title="Scaled mean\nexpression",
        size_title="Pct. expressed",
        return_fig=True,
        figsize=(8.5, 3.5),
    )
    dp.make_figure()
    main_ax = dp.ax_dict["mainplot_ax"]
    main_ax.set_xticklabels(main_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    main_ax.tick_params(axis="x", labelsize=9)
    main_ax.tick_params(axis="y", labelsize=10)
    fig = plt.gcf()
    fig.savefig(fig_dir / f"dotplot_{GROUP_LABEL_MAP[group_name].lower()}_common_aqueous_up_v01.pdf", bbox_inches="tight")
    plt.close(fig)


parser = ArgumentParser()
parser.add_argument("--deg-dir", type=Path, required=True)
parser.add_argument("--gsea-calc-dir", type=Path, required=True)
parser.add_argument("--fig-dir", type=Path, required=True)
parser.add_argument("--input-h5ad", type=Path, required=True)
parser.add_argument("--hallmark-gmt", type=Path, required=True)
parser.add_argument("--reactome-gmt", type=Path, required=True)
parser.add_argument("--gobp-gmt", type=Path, required=True)
parser.add_argument("--label-col", default="predicted_labels_Immune_All_Low_leiden_r05")
parser.add_argument("--tissue-col", default="Tissue")
parser.add_argument("--cd8-group", default="CD8_4labels")
parser.add_argument("--myeloid-group", default="Monocyte_like_3labels")
parser.add_argument("--resample-tag", default="resamp100")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--fdr-cut", type=float, default=0.05)
parser.add_argument("--effect-cut", type=float, default=0.25)
args = parser.parse_args()

args.gsea_calc_dir.mkdir(parents=True, exist_ok=True)
args.fig_dir.mkdir(parents=True, exist_ok=True)

for fp in [args.hallmark_gmt, args.reactome_gmt, args.gobp_gmt, args.input_h5ad]:
    if not fp.exists():
        raise FileNotFoundError(fp)

group_names = [args.cd8_group, args.myeloid_group]
for group_name in group_names:
    if group_name not in ANALYSIS_LABELS:
        raise ValueError(f"Unsupported group: {group_name}")

gene_set_files = {
    "hallmark": args.hallmark_gmt,
    "reactome": args.reactome_gmt,
    "gobp": args.gobp_gmt,
}

deg_file_map = {}
for group_name in group_names:
    deg_file_map[group_name] = {
        "Aqueous_vs_PBMC": args.deg_dir / f"DEG_{group_name}_aqueous_vs_pbmc_log1p_{args.resample_tag}_v01.tsv",
        "Aqueous_vs_CSF": args.deg_dir / f"DEG_{group_name}_aqueous_vs_csf_log1p_{args.resample_tag}_v01.tsv",
    }
    for fp in deg_file_map[group_name].values():
        if not fp.exists():
            raise FileNotFoundError(fp)

adata = sc.read_h5ad(args.input_h5ad)
all_hallmark = []

for group_name, comp_map in deg_file_map.items():
    for comparison, deg_fp in comp_map.items():
        deg_df = read_deg_tsv(deg_fp)
        prerank_df = make_prerank_df(deg_df)
        for collection_name, gs_fp in gene_set_files.items():
            out_fp = args.gsea_calc_dir / f"GSEA_{collection_name}_{group_name}_{comparison}_log1p_{args.resample_tag}_v01.tsv"
            res2d = run_prerank_and_save(prerank_df, gs_fp, out_fp, threads=args.threads)
            res2d = standardize_gsea_df(res2d, group_name, comparison, collection_name)
            if collection_name == "hallmark":
                all_hallmark.append(res2d)

plot_df = pd.concat(all_hallmark, axis=0, ignore_index=True)
plot_df.to_csv(args.gsea_calc_dir / f"GSEA_hallmark_summary_{args.resample_tag}_v01.tsv", sep="\t", index=False)

for group_name in group_names:
    save_hallmark_barplot(plot_df, group_name, args.fig_dir)
    save_common_gene_panel(
        group_name=group_name,
        deg_files=deg_file_map[group_name],
        adata=adata,
        fig_dir=args.fig_dir,
        calc_dir=args.gsea_calc_dir,
        label_col=args.label_col,
        tissue_col=args.tissue_col,
        effect_cut=args.effect_cut,
        fdr_cut=args.fdr_cut,
    )

save_hallmark_bubble(plot_df, group_names, args.fig_dir)
