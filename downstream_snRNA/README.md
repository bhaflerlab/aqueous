# Downstream snRNA-seq analysis

This folder contains downstream single-cell analysis scripts for the aqueous humor project.

## Contents

- `scripts/`: analysis scripts
- `run/`: run scripts and rerun commands

## Revised CD8 pseudobulk and postmortem sensitivity analysis

Last updated: 2026-11-13 US Eastern Time.

The revised CD8 analysis uses donor-level raw-count sums and the model
`~ age_centered + sex + condition`. The existing `03_deg_resample.py` and
`04_gsea.py` preserve the earlier cell-resampling workflow; use the three dated
scripts below for the revised CD8 results.

Run commands from this `downstream_snRNA` directory, with a new output directory
for each step. The scripts do not overwrite existing output directories.

### Inputs

- An annotated, full-gene H5AD from the integration/annotation workflow, with
  integer raw counts in `layers/counts`, gene symbols in `var_names`, and
  `Tissue`, `sample`, `SampleID`, and
  `predicted_labels_Immune_All_Low_leiden_r05` in `obs`.
- A curated tab-separated sample roster with columns `unit_id`, `Tissue`,
  `sample`, `SampleID`, `donor_id`, `age`, and `sex`. `Tissue` is `aqueous`,
  `pbmc`, or `csf`; `age` is continuous in years; `sex` is `F` or `M`.
  Aqueous rows match `obs.sample`; reference rows match `obs.SampleID`.
  The unused identifier column may be empty. Use independent, resolved aqueous
  donors and healthy reference samples; pooled/hashed aqueous samples are excluded.
  The manuscript roster has four aqueous donors and twelve donors per reference
  tissue; PBMC and CSF share the reference donors and are analyzed separately.
- Ferreira et al. (2018), Supplementary Data 2:
  [whole-blood pre/postmortem effects workbook](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-017-02772-x/MediaObjects/41467_2017_2772_MOESM5_ESM.xlsx).
  The script reads `DiffExp_Blood_Pre_vs_Postm` with the second row as its header.
- MSigDB Hallmark `h.all.v2026.1.Hs.symbols.gmt`, obtained from MSigDB under its
  access terms. Its SHA-256 in the manuscript analysis is
  `eecaf6dad908334ae885406ec72bdc0646d8917588ed7c219fac92fc5363f596`.

The donor roster, raw expression data, and GMT are supplied separately; they are
not embedded in these scripts.

### Run

```bash
python scripts/261113_prepare_cd8_pseudobulk.py \
  --input-h5ad /path/to/annotated_fullgene.h5ad \
  --sample-metadata /path/to/curated_samples.tsv \
  --output-dir /path/to/new_cd8_counts

python scripts/261113_cd8_age_sex_pseudobulk.py \
  --counts /path/to/new_cd8_counts/261113_cd8_counts.tsv.gz \
  --sample-metadata /path/to/new_cd8_counts/261113_cd8_sample_metadata.tsv \
  --output-dir /path/to/new_cd8_pseudobulk --threads 4

python scripts/261113_cd8_postmortem_gsea.py \
  --deg-dir /path/to/new_cd8_pseudobulk \
  --ferreira-supplement /path/to/41467_2017_2772_MOESM5_ESM.xlsx \
  --hallmark-gmt /path/to/h.all.v2026.1.Hs.symbols.gmt \
  --output-dir /path/to/new_cd8_gsea --threads 4
```

### Analysis settings

CD8 cells comprise the four cytotoxic labels listed explicitly in the preparation
script. Counts are summed within sample units without normalization. Samples
require at least five CD8 cells; genes require at least ten raw counts in at least
three samples within each contrast. PyDESeq2 uses median-of-ratios size factors,
parametric dispersion fitting, default Cook filtering/refitting, and independent
filtering. Age is centered within each contrast. Unshrunk condition Wald estimates
are retained, with `inference_eligible=False` for nonconverged coefficients.

GSEA ranks every inference-eligible gene with a finite Wald statistic and p-value;
no DEG significance filter is applied. The original GSEA uses the complete ranking.
The paired sensitivity analysis removes only the 2,339 genes meeting external
FDR <0.01 and absolute log2 fold change >=1. Non-text spreadsheet symbols and all
ambiguous duplicate symbols are omitted before defining this set. Genes untested
by Ferreira et al. remain in the ranking. This reproduces the direct-exclusion
analysis, without restricting the background to GTEx-tested genes.

GSEA uses 1,000 gene-set permutations, seed 1729, weight 1, and gene-set sizes
15–500. FDR is calculated within eligible Hallmark sets for each ranking. The output
includes all pathway results, the three pathways displayed in the sensitivity
figure, removed genes, symbol exclusions, input checksums, and software versions.
Gene exclusion is a sensitivity analysis; it does not establish independence from
postmortem effects. The four aqueous donors, limited age overlap, and tissue/study
confounding remain limitations.

Tested software: Python 3.11, PyDESeq2 0.5.3, GSEApy 1.1.11, NumPy 2.3.5,
pandas 2.3.3, SciPy 1.15.3, AnnData 0.12.7, h5py 3.15.1, and openpyxl 3.1.5.

Reference: Ferreira PG et al. *The effects of death and post-mortem cold ischemia
on human tissue transcriptomes.* Nature Communications 9, 490 (2018).
[doi:10.1038/s41467-017-02772-x](https://doi.org/10.1038/s41467-017-02772-x).
