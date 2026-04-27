#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def load_input(in_path: Path, trait_name: str) -> tuple[pd.DataFrame, int]:
    if trait_name == "au":
        df = pd.read_csv(in_path, sep="\t", low_memory=False)
        n_total = int(df["N"].iloc[0])
        return df[["SNP", "CHR", "BP", "P"]].copy(), n_total

    if trait_name == "amd":
        df = pd.read_csv(in_path, sep="\t", low_memory=False)
        if {"SNP", "CHR", "BP", "P", "N"}.issubset(df.columns):
            n_total = int(df["N"].iloc[0])
            return df[["SNP", "CHR", "BP", "P"]].copy(), n_total
        df = df.rename(columns={"Marker": "SNP", "Chrom": "CHR", "Pos": "BP", "GC.Pvalue": "P"})
        n_total = int(df["Ncases"].iloc[0] + df["Ncontrols"].iloc[0])
        return df[["SNP", "CHR", "BP", "P"]].copy(), n_total

    if trait_name == "poag":
        df = pd.read_csv(in_path, sep="\t", low_memory=False)
        if {"SNP", "CHR", "BP", "P", "N"}.issubset(df.columns):
            n_total = int(df["N"].iloc[0])
            return df[["SNP", "CHR", "BP", "P"]].copy(), n_total
        fallback = (
            df["#chrom"].astype(str)
            + ":"
            + df["pos"].astype(str)
            + ":"
            + df["ref"].astype(str)
            + ":"
            + df["alt"].astype(str)
        )
        snp = df["rsids"].astype(str)
        df["SNP"] = snp.where((snp != "") & (snp != ".") & (snp != "nan"), fallback)
        df = df.rename(columns={"#chrom": "CHR", "pos": "BP", "pval": "P"})
        n_total = 7756 + 358375
        return df[["SNP", "CHR", "BP", "P"]].copy(), n_total

    raise ValueError(f"Unknown trait_name: {trait_name}")


def write_gs(gene_pval_path: Path, gs_path: Path, trait_id: str) -> None:
    df = pd.read_csv(gene_pval_path, sep="\t")
    score_col = df.columns[1]
    df = df.dropna(subset=["GENE", score_col]).copy()
    df = df[df[score_col] > 0].copy()
    df["weight"] = -np.log10(df[score_col].astype(float))
    df = df[df["weight"] > 0].sort_values("weight", ascending=False)
    geneset = ",".join(f"{gene}:{weight:.4f}" for gene, weight in zip(df["GENE"], df["weight"]))
    with open(gs_path, "w") as fout:
        fout.write("TRAIT\tGENESET\n")
        fout.write(f"{trait_id}\t{geneset}\n")


def load_gene_loc_map(gene_loc: Path) -> pd.DataFrame:
    cols = ["ENTREZ", "CHR", "START", "STOP", "STRAND", "GENE"]
    df = pd.read_csv(gene_loc, sep="\t", header=None, names=cols)
    df["ENTREZ"] = df["ENTREZ"].astype(str)
    return df[["ENTREZ", "GENE"]].drop_duplicates()


parser = ArgumentParser()
parser.add_argument("--trait-name", required=True, choices=["au", "amd", "poag"])
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--trait-id", required=True)
parser.add_argument("--outdir", type=Path, required=True)
parser.add_argument("--magma-bin", type=Path, required=True)
parser.add_argument("--magma-bfile", type=Path, required=True)
parser.add_argument("--gene-loc", type=Path, required=True)
args = parser.parse_args()

args.outdir.mkdir(parents=True, exist_ok=True)
magma_prefix = args.outdir / f"{args.trait_id}_magma"
gene_pval_path = args.outdir / f"{args.trait_id}_magma_gene_pval.tsv"
gs_path = args.outdir / f"{args.trait_id}_scdrs.gs"

df, n_total = load_input(args.input, args.trait_name)
df = df.dropna(subset=["SNP", "CHR", "BP", "P"]).copy()
df["CHR"] = df["CHR"].astype(str).str.replace("^chr", "", regex=True)
df["BP"] = pd.to_numeric(df["BP"], errors="coerce").astype(int)
df["P"] = pd.to_numeric(df["P"], errors="coerce")
df = df[(df["P"] > 0) & (df["P"] <= 1)].copy()

genes_out = magma_prefix.with_suffix(".genes.out")
genes_annot = magma_prefix.with_suffix(".genes.annot")
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    snp_loc_path = tmpdir / "snp_loc.tsv"
    pval_path = tmpdir / "pval.tsv"
    df[["SNP", "CHR", "BP"]].to_csv(snp_loc_path, sep="\t", index=False, header=False)
    df[["SNP", "P"]].to_csv(pval_path, sep="\t", index=False, header=True)

    run_cmd(
        [
            str(args.magma_bin),
            "--annotate",
            "--snp-loc",
            str(snp_loc_path),
            "--gene-loc",
            str(args.gene_loc),
            "--out",
            str(magma_prefix),
        ]
    )
    run_cmd(
        [
            str(args.magma_bin),
            "--bfile",
            str(args.magma_bfile),
            "--pval",
            str(pval_path),
            f"N={n_total}",
            "--gene-annot",
            str(genes_annot),
            "--out",
            str(magma_prefix),
            "--genes-only",
        ]
    )

df_gene = pd.read_csv(genes_out, sep=r"\s+", comment="#")
df_gene["GENE"] = df_gene["GENE"].astype(str)
df_gene = df_gene.rename(columns={"GENE": "ENTREZ", "P": args.trait_id})
df_gene = df_gene.merge(load_gene_loc_map(args.gene_loc), on="ENTREZ", how="left")
df_gene["GENE"] = df_gene["GENE"].fillna(df_gene["ENTREZ"])
df_gene[["GENE", args.trait_id]].to_csv(gene_pval_path, sep="\t", index=False)
write_gs(gene_pval_path, gs_path, args.trait_id)

print(gene_pval_path)
print(gs_path)
