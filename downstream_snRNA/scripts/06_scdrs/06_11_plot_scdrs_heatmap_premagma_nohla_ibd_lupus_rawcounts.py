#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_scdrs")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_scdrs")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scdrs

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


TISSUE_ORDER = ["AQ", "PBMC", "CSF"]
TISSUE_SOURCE_ORDER = ["aqueous", "pbmc", "csf"]
TISSUE_SOURCE_TO_LABEL = {"aqueous": "AQ", "pbmc": "PBMC", "csf": "CSF"}
CELLTYPE_ORDER = ["CD8", "CD4", "Treg", "NK/NKT/ILC", "B/plasma", "Mono/Mac", "DC"]
CELLTYPE_SOURCE_ORDER = ["CD8", "CD4", "Treg", "NK_NKT_ILC", "B_plasma", "Monocyte_Macrophage", "DC"]
CELLTYPE_SOURCE_TO_LABEL = {
    "CD8": "CD8",
    "CD4": "CD4",
    "Treg": "Treg",
    "NK_NKT_ILC": "NK/NKT/ILC",
    "B_plasma": "B/plasma",
    "Monocyte_Macrophage": "Mono/Mac",
    "DC": "DC",
}


def reshape_rows(df: pd.DataFrame, trait_specs: list[dict[str, Path | str]]) -> pd.DataFrame:
    rows = []
    for tissue in TISSUE_ORDER:
        for spec in trait_specs:
            row_name = f"{spec['label']} | {tissue}"
            rows.append(pd.Series({ct: df.loc[spec["label"], f"{tissue} | {ct}"] for ct in CELLTYPE_ORDER}, name=row_name))
    return pd.DataFrame(rows)


parser = ArgumentParser()
parser.add_argument("--h5ad", type=Path, required=True)
parser.add_argument("--au-score-dir", type=Path, required=True)
parser.add_argument("--amd-score-dir", type=Path, required=True)
parser.add_argument("--poag-score-dir", type=Path, required=True)
parser.add_argument("--control-score-dir", type=Path, required=True)
parser.add_argument("--downstream-dir", type=Path, required=True)
parser.add_argument("--fig-dir", type=Path, required=True)
parser.add_argument("--out-prefix", required=True)
args = parser.parse_args()

args.downstream_dir.mkdir(parents=True, exist_ok=True)
args.fig_dir.mkdir(parents=True, exist_ok=True)

trait_specs: list[dict[str, Path | str]] = [
    {"score_dir": args.au_score_dir, "trait": "GCST90295958_preMagmaNoHLA_AU", "label": "AU preMagma noHLA"},
    {"score_dir": args.amd_score_dir, "trait": "Fritsche_2015_AdvancedAMD_preMagmaNoHLA", "label": "AMD preMagma noHLA"},
    {"score_dir": args.poag_score_dir, "trait": "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA", "label": "POAG preMagma noHLA"},
    {"score_dir": args.control_score_dir, "trait": "PASS_Lupus_noHLA", "label": "Lupus noHLA"},
    {"score_dir": args.control_score_dir, "trait": "PASS_IBD_deLange2017_noHLA", "label": "IBD noHLA"},
]

adata = ad.read_h5ad(args.h5ad, backed="r").to_memory()
adata.obs["tissue_broad"] = adata.obs["Tissue"].astype(str) + " | " + adata.obs["broad_celltype"].astype(str)

group_order = []
group_display = {}
for tissue in TISSUE_SOURCE_ORDER:
    for celltype in CELLTYPE_SOURCE_ORDER:
        group = f"{tissue} | {celltype}"
        if (adata.obs["tissue_broad"] == group).any():
            group_order.append(group)
            group_display[group] = f"{TISSUE_SOURCE_TO_LABEL[tissue]} | {CELLTYPE_SOURCE_TO_LABEL[celltype]}"

dict_df_stats = {}
for spec in trait_specs:
    df_score = pd.read_csv(Path(spec["score_dir"]) / f"{spec['trait']}.full_score.gz", sep="\t", index_col=0)
    df_stats = scdrs.method.downstream_group_analysis(
        adata=adata,
        df_full_score=df_score,
        group_cols=["tissue_broad"],
    )["tissue_broad"].reindex(group_order)
    dict_df_stats[spec["trait"]] = df_stats
    df_stats.rename(index=group_display).to_csv(
        args.downstream_dir / f"{spec['trait']}.scdrs_group.tissue_broad.tsv",
        sep="\t",
    )

df_fdr = pd.concat(
    [((dict_df_stats[spec["trait"]]["n_fdr_0.1"] / dict_df_stats[spec["trait"]]["n_cell"]).rename(spec["label"])) for spec in trait_specs],
    axis=1,
).T.rename(columns=group_display)
df_assoc = pd.concat(
    [(dict_df_stats[spec["trait"]]["assoc_mcp"].rename(spec["label"])) for spec in trait_specs],
    axis=1,
).T.rename(columns=group_display)
df_hetero = pd.concat(
    [(dict_df_stats[spec["trait"]]["hetero_mcp"].rename(spec["label"])) for spec in trait_specs],
    axis=1,
).T.rename(columns=group_display)

plot_fdr = reshape_rows(df_fdr, trait_specs)
plot_assoc = reshape_rows(df_assoc, trait_specs)
plot_hetero = reshape_rows(df_hetero, trait_specs)

fig, ax = scdrs.util.plot_group_stats(
    df_fdr_prop=plot_fdr,
    df_assoc_fdr=plot_assoc,
    df_hetero_fdr=plot_hetero,
    plot_kws={
        "square_size": 45,
        "cb_fraction": 0.035,
        "cb_pad": 0.02,
        "cb_vmax": 0.2,
        "hetero_size": 6,
        "signif_size": 0.7,
    },
)
ax.set_xlabel("")
ax.set_ylabel("")
for y in [len(trait_specs), len(trait_specs) * 2]:
    ax.hlines(y, *ax.get_xlim(), colors="black", linewidth=0.8, linestyles=(0, (2, 2)))

fig.savefig(args.fig_dir / f"{args.out_prefix}.pdf", bbox_inches="tight")
plt.close(fig)
plot_fdr.to_csv(args.fig_dir / f"{args.out_prefix}_fdr_prop.tsv", sep="\t")
plot_assoc.to_csv(args.fig_dir / f"{args.out_prefix}_assoc_fdr.tsv", sep="\t")
plot_hetero.to_csv(args.fig_dir / f"{args.out_prefix}_hetero_fdr.tsv", sep="\t")

print(args.fig_dir / f"{args.out_prefix}.pdf")
