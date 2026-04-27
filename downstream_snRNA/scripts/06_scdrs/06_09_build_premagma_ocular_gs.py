#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


parser = ArgumentParser()
parser.add_argument("--out-base", type=Path, required=True)
parser.add_argument("--amd-input", type=Path, required=True)
parser.add_argument("--poag-input", type=Path, required=True)
parser.add_argument("--au-input", type=Path, required=True)
parser.add_argument("--chain", type=Path, required=True)
parser.add_argument("--bim", type=Path, required=True)
parser.add_argument("--magma-bin", type=Path, required=True)
parser.add_argument("--magma-bfile", type=Path, required=True)
parser.add_argument("--gene-loc", type=Path, required=True)
parser.add_argument("--liftover-python", type=Path, default=Path(sys.executable))
parser.add_argument("--traits", default="au,amd,poag")
args = parser.parse_args()

selected_traits = {x.strip().lower() for x in args.traits.split(",") if x.strip()}
trait_dir = args.out_base / "magma"
trait_dir.mkdir(parents=True, exist_ok=True)

manifest_rows = []
gs_rows = []

if "amd" in selected_traits:
    amd_scdrs = trait_dir / "Fritsche_2015_AdvancedAMD_scdrs_hg19.tsv.gz"
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_01_format_amd_for_scdrs.py",
            "--input",
            args.amd_input,
            "--output",
            amd_scdrs,
        ]
    )
    amd_nohla = trait_dir / "Fritsche_2015_AdvancedAMD_preMagmaNoHLA_sumstats.tsv.gz"
    amd_report = trait_dir / "Fritsche_2015_AdvancedAMD_preMagmaNoHLA_sumstats_hla_filter_report.tsv"
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_07_filter_hla_region_from_sumstats.py",
            "--input",
            amd_scdrs,
            "--output",
            amd_nohla,
            "--report",
            amd_report,
        ]
    )
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_08_run_magma_make_gs.py",
            "--trait-name",
            "amd",
            "--input",
            amd_nohla,
            "--trait-id",
            "Fritsche_2015_AdvancedAMD_preMagmaNoHLA",
            "--outdir",
            trait_dir / "Fritsche_2015_AdvancedAMD_preMagmaNoHLA",
            "--magma-bin",
            args.magma_bin,
            "--magma-bfile",
            args.magma_bfile,
            "--gene-loc",
            args.gene_loc,
        ]
    )
    amd_gs = trait_dir / "Fritsche_2015_AdvancedAMD_preMagmaNoHLA" / "Fritsche_2015_AdvancedAMD_preMagmaNoHLA_scdrs.gs"
    gs_rows.append(pd.read_csv(amd_gs, sep="\t"))
    manifest_rows.append({"trait_name": "amd", "trait_id": "Fritsche_2015_AdvancedAMD_preMagmaNoHLA", "input": str(args.amd_input), "magma_input": str(amd_nohla), "gs": str(amd_gs)})

if "poag" in selected_traits:
    poag_hg19 = trait_dir / "finngen_R9_H7_GLAUCOMA_POAG_hg19.tsv.gz"
    poag_liftover_report = trait_dir / "finngen_R9_H7_GLAUCOMA_POAG_hg19_liftover_report.tsv"
    poag_scdrs = trait_dir / "finngen_R9_H7_GLAUCOMA_POAG_scdrs_hg19.tsv.gz"
    run_cmd(
        [
            args.liftover_python,
            SCRIPT_DIR / "06_02_liftover_poag_to_hg19.py",
            "--input",
            args.poag_input,
            "--chain",
            args.chain,
            "--output",
            poag_hg19,
            "--report",
            poag_liftover_report,
        ]
    )
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_03_format_poag_for_scdrs.py",
            "--input",
            poag_hg19,
            "--output",
            poag_scdrs,
        ]
    )
    poag_nohla = trait_dir / "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA_sumstats.tsv.gz"
    poag_report = trait_dir / "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA_sumstats_hla_filter_report.tsv"
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_07_filter_hla_region_from_sumstats.py",
            "--input",
            poag_scdrs,
            "--output",
            poag_nohla,
            "--report",
            poag_report,
        ]
    )
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_08_run_magma_make_gs.py",
            "--trait-name",
            "poag",
            "--input",
            poag_nohla,
            "--trait-id",
            "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA",
            "--outdir",
            trait_dir / "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA",
            "--magma-bin",
            args.magma_bin,
            "--magma-bfile",
            args.magma_bfile,
            "--gene-loc",
            args.gene_loc,
        ]
    )
    poag_gs = trait_dir / "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA" / "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA_scdrs.gs"
    gs_rows.append(pd.read_csv(poag_gs, sep="\t"))
    manifest_rows.append({"trait_name": "poag", "trait_id": "finngen_R9_H7_GLAUCOMA_POAG_preMagmaNoHLA", "input": str(args.poag_input), "magma_input": str(poag_nohla), "gs": str(poag_gs)})

if "au" in selected_traits:
    au_scdrs_hg38 = trait_dir / "GCST90295958_sumstats_hg38_scdrs.tsv.gz"
    au_scdrs_hg38_report = trait_dir / "GCST90295958_sumstats_hg38_scdrs_report.tsv"
    au_hg19 = trait_dir / "GCST90295958_sumstats_hg19.tsv.gz"
    au_liftover_report = trait_dir / "GCST90295958_sumstats_hg19_liftover_report.tsv"
    au_nohla = trait_dir / "GCST90295958_preMagmaNoHLA_sumstats.tsv.gz"
    au_nohla_report = trait_dir / "GCST90295958_preMagmaNoHLA_sumstats_hla_filter_report.tsv"
    au_magma = trait_dir / "GCST90295958_preMagmaNoHLA_sumstats_magma.tsv.gz"
    au_magma_report = trait_dir / "GCST90295958_preMagmaNoHLA_sumstats_magma_report.tsv"
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_04_format_au_from_gwas_catalog.py",
            "--input",
            args.au_input,
            "--output",
            au_scdrs_hg38,
            "--report",
            au_scdrs_hg38_report,
        ]
    )
    run_cmd(
        [
            args.liftover_python,
            SCRIPT_DIR / "06_05_liftover_au_to_hg19.py",
            "--input",
            au_scdrs_hg38,
            "--chain",
            args.chain,
            "--output",
            au_hg19,
            "--report",
            au_liftover_report,
        ]
    )
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_07_filter_hla_region_from_sumstats.py",
            "--input",
            au_hg19,
            "--output",
            au_nohla,
            "--report",
            au_nohla_report,
        ]
    )
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_06_map_au_to_reference_rsid.py",
            "--input",
            au_nohla,
            "--bim",
            args.bim,
            "--output",
            au_magma,
            "--report",
            au_magma_report,
        ]
    )
    run_cmd(
        [
            sys.executable,
            SCRIPT_DIR / "06_08_run_magma_make_gs.py",
            "--trait-name",
            "au",
            "--input",
            au_magma,
            "--trait-id",
            "GCST90295958_preMagmaNoHLA_AU",
            "--outdir",
            trait_dir / "GCST90295958_preMagmaNoHLA_AU",
            "--magma-bin",
            args.magma_bin,
            "--magma-bfile",
            args.magma_bfile,
            "--gene-loc",
            args.gene_loc,
        ]
    )
    au_gs = trait_dir / "GCST90295958_preMagmaNoHLA_AU" / "GCST90295958_preMagmaNoHLA_AU_scdrs.gs"
    gs_rows.append(pd.read_csv(au_gs, sep="\t"))
    manifest_rows.append({"trait_name": "au", "trait_id": "GCST90295958_preMagmaNoHLA_AU", "input": str(args.au_input), "magma_input": str(au_magma), "gs": str(au_gs)})

combined = pd.concat(gs_rows, axis=0, ignore_index=True)
combined_gs = args.out_base / "ocular_traits_premagma_nohla.gs"
manifest_path = args.out_base / "ocular_traits_premagma_nohla_manifest.tsv"
combined.to_csv(combined_gs, sep="\t", index=False)
pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)

print(combined_gs)
print(manifest_path)
