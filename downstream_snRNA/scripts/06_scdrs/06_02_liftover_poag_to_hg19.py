#!/usr/bin/env python3

from argparse import ArgumentParser
import csv
import gzip
from pathlib import Path

from pyliftover import LiftOver


def normalize_chr(chrom: str) -> str:
    value = str(chrom).strip()
    if value.startswith("chr"):
        value = value[3:]
    if value == "M":
        return "MT"
    return value


parser = ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--chain", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()

lo = LiftOver(str(args.chain))
total_rows = 0
mapped_rows = 0
unmapped_rows = 0
multi_map_rows = 0
alt_chr_rows = 0

args.output.parent.mkdir(parents=True, exist_ok=True)
args.report.parent.mkdir(parents=True, exist_ok=True)

with gzip.open(args.input, "rt") as fin, gzip.open(args.output, "wt") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()

    for row in reader:
        total_rows += 1
        chrom = normalize_chr(row["#chrom"])
        pos_1based = int(row["pos"])
        hits = lo.convert_coordinate(f"chr{chrom}", pos_1based - 1)

        if not hits:
            unmapped_rows += 1
            continue

        if len(hits) > 1:
            multi_map_rows += 1

        new_chrom, new_pos_0based, _, _ = hits[0]
        new_chrom = normalize_chr(new_chrom)
        if new_chrom != chrom:
            alt_chr_rows += 1

        row["#chrom"] = new_chrom
        row["pos"] = str(int(new_pos_0based) + 1)
        writer.writerow(row)
        mapped_rows += 1

with open(args.report, "w") as fout:
    fout.write("metric\tvalue\n")
    fout.write(f"input_file\t{args.input}\n")
    fout.write(f"chain_file\t{args.chain}\n")
    fout.write(f"output_file\t{args.output}\n")
    fout.write(f"total_rows\t{total_rows}\n")
    fout.write(f"mapped_rows\t{mapped_rows}\n")
    fout.write(f"unmapped_rows\t{unmapped_rows}\n")
    fout.write(f"multi_map_rows\t{multi_map_rows}\n")
    fout.write(f"alt_chr_rows\t{alt_chr_rows}\n")
    fout.write(f"mapped_fraction\t{mapped_rows / total_rows:.8f}\n")

print(args.output)
print(args.report)
