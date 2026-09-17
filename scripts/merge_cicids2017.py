#!/usr/bin/env python3
"""
merge_cicids2017.py

Safely merge the 8 original CICIDS2017 "TrafficLabelling" CSV files into ONE
raw merged CSV, for thesis reproducibility.

Design / safety guarantees
---------------------------
* The 8 original CSV files are NEVER modified, renamed, cleaned, encoded,
  downsampled, deduplicated or preprocessed. They are opened read-only.
* Vertical (row-wise) concatenation only. No database join.
* Chronological order is fixed explicitly (see ORDER below).
* Only the COLUMN NAMES are stripped of accidental leading/trailing whitespace
  (e.g. " Label" -> "Label") and of a leading UTF-8 BOM. Data VALUES are never
  altered.
* Reading and writing both use latin-1. latin-1 is a lossless 1:1 byte<->char
  codec, so every original data byte (including the non-UTF8 0x96 en-dash byte
  found inside some Thursday-WebAttacks labels) is preserved BYTE-FOR-BYTE in
  the merged output. The `Label` values are therefore untouched.
* A single provenance column `source_file` is appended (original filename).
* Chunked streaming (CHUNK_ROWS) so the full dataset is never held in RAM.
* If any file's schema (stripped column names) differs from the first file,
  the script STOPS and reports exactly which file/columns differ, merging
  nothing.

No missing-value handling, infinity handling, duplicate removal, balancing,
feature selection, encoding, scaling or splitting is performed.
"""

import csv
import os
import sys

# --- configuration ----------------------------------------------------------

RAW_DIR = r"E:\Shameem\thesis\data\raw\CIC-IDS-2017\TrafficLabelling"
OUT_DIR = r"E:\Shameem\thesis\data\processed\CIC-IDS-2017"
OUT_FILE = os.path.join(OUT_DIR, "CICIDS2017_merged_raw.csv")

# Chronological order (rule 4). These are the "logical" order keys; the actual
# on-disk filenames are matched case-insensitively to avoid assuming exact
# capitalization/spelling.
ORDER = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos",
]

CHUNK_ROWS = 100_000
ENCODING = "latin-1"  # lossless byte-preserving read/write
PROVENANCE_COL = "source_file"
BOM = "\ufeff"


def die(msg: str) -> "None":
    print("\n*** STOP: " + msg, file=sys.stderr)
    sys.exit(1)


def clean_header(raw_header):
    """Strip a leading BOM and strip whitespace from each column NAME only."""
    cleaned = []
    for i, name in enumerate(raw_header):
        if i == 0:
            # BOM may appear as '\ufeff' (utf-8-sig) or as latin-1 bytes 'ï»¿'.
            name = name.lstrip(BOM)
            if name.startswith("\xef\xbb\xbf"):
                name = name[3:]
        cleaned.append(name.strip())
    return cleaned


def match_files(raw_dir):
    """Match each ORDER key to a real on-disk filename (case-insensitive)."""
    present = os.listdir(raw_dir)
    lower_map = {}
    for fn in present:
        lower_map.setdefault(fn.lower(), fn)

    matched = []
    for key in ORDER:
        # each real CICIDS2017 file is "<key>.pcap_ISCX.csv"
        hits = [fn for fn in present
                if fn.lower().startswith(key.lower()) and fn.lower().endswith(".csv")]
        if len(hits) == 0:
            die(f"could not find a CSV file for order key '{key}' in {raw_dir}")
        if len(hits) > 1:
            die(f"ambiguous match for order key '{key}': {hits}")
        matched.append(hits[0])
    return present, matched


def read_header(path):
    with open(path, "r", encoding=ENCODING, newline="") as fh:
        reader = csv.reader(fh)
        return clean_header(next(reader))


def main():
    print("=" * 70)
    print("CICIDS2017 raw merge")
    print("=" * 70)

    if not os.path.isdir(RAW_DIR):
        die(f"raw directory not found: {RAW_DIR}")

    present, matched = match_files(RAW_DIR)
    csv_present = [f for f in present if f.lower().endswith(".csv")]
    print(f"CSV files found in directory : {len(csv_present)}")
    print(f"Files matched to merge order : {len(matched)}")
    print("\nMerge order (chronological):")
    for i, fn in enumerate(matched, 1):
        print(f"  {i}. {fn}")

    # --- schema verification (rule 6/8) -------------------------------------
    ref_header = read_header(os.path.join(RAW_DIR, matched[0]))
    ref_ncols = len(ref_header)
    schemas_match = True
    print(f"\nColumn count (before adding '{PROVENANCE_COL}'): {ref_ncols}")

    for fn in matched:
        hdr = read_header(os.path.join(RAW_DIR, fn))
        if hdr != ref_header:
            schemas_match = False
            print(f"\nSCHEMA MISMATCH in {fn}:")
            if len(hdr) != ref_ncols:
                print(f"  column count {len(hdr)} != reference {ref_ncols}")
            for idx, (a, b) in enumerate(zip(ref_header, hdr)):
                if a != b:
                    print(f"  col {idx}: reference={a!r} got={b!r}")
            extra = hdr[len(ref_header):]
            miss = ref_header[len(hdr):]
            if extra:
                print(f"  extra columns: {extra}")
            if miss:
                print(f"  missing columns: {miss}")

    if not schemas_match:
        die("schemas do not match across files; nothing was merged.")
    print("All schemas matched            : YES")

    label_idx = ref_header.index("Label") if "Label" in ref_header else None
    if label_idx is None:
        die("no 'Label' column found after stripping names.")

    out_header = ref_header + [PROVENANCE_COL]

    # --- streaming merge (rule 3/10) ----------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    per_file_rows = {}
    label_counts = {}
    total_source_rows = 0
    empty_rows_skipped = 0

    print(f"\nWriting merged output -> {OUT_FILE}")
    with open(OUT_FILE, "w", encoding=ENCODING, newline="") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(out_header)

        for fn in matched:
            path = os.path.join(RAW_DIR, fn)
            n_rows = 0
            buffer = []
            with open(path, "r", encoding=ENCODING, newline="") as in_fh:
                reader = csv.reader(in_fh)
                next(reader)  # skip header
                for row in reader:
                    if not row:  # trailing blank line, not a data row
                        empty_rows_skipped += 1
                        continue
                    label = row[label_idx] if label_idx < len(row) else ""
                    label_counts[label] = label_counts.get(label, 0) + 1
                    buffer.append(row + [fn])
                    n_rows += 1
                    if len(buffer) >= CHUNK_ROWS:
                        writer.writerows(buffer)
                        buffer.clear()
                if buffer:
                    writer.writerows(buffer)
                    buffer.clear()
            per_file_rows[fn] = n_rows
            total_source_rows += n_rows
            print(f"  merged {fn:<55} rows={n_rows:,}")

    # --- validation (rule: report everything) -------------------------------
    # Count rows actually present in the merged file (independent recount).
    merged_rows = 0
    with open(OUT_FILE, "r", encoding=ENCODING, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if row:
                merged_rows += 1

    out_size = os.path.getsize(OUT_FILE)

    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    print(f"CSV files found                       : {len(csv_present)}")
    print(f"Files merged                          : {len(matched)}")
    print(f"Column count before '{PROVENANCE_COL}'        : {ref_ncols}")
    print(f"Column count after  '{PROVENANCE_COL}'        : {len(out_header)}")
    print(f"All schemas matched                   : {schemas_match}")
    if empty_rows_skipped:
        print(f"Trailing empty lines skipped          : {empty_rows_skipped}")

    print("\nRow count of each original file:")
    for fn in matched:
        print(f"  {fn:<55} {per_file_rows[fn]:>12,}")

    print(f"\nTotal source rows (sum of 8 files)    : {total_source_rows:,}")
    print(f"Total rows in merged CSV              : {merged_rows:,}")
    ok = total_source_rows == merged_rows
    print(f"sum(source rows) == merged rows       : {ok}")

    print("\nRows per source_file (merged dataset):")
    for fn in matched:
        print(f"  {fn:<55} {per_file_rows[fn]:>12,}")

    print("\nLabel distribution (merged dataset):")
    for label, cnt in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<40} {cnt:>12,}")

    print(f"\nOutput file : {OUT_FILE}")
    print(f"File size   : {out_size:,} bytes ({out_size / (1024*1024):.1f} MB)")

    if not ok:
        die("row-count mismatch between source total and merged file.")
    print("\nMERGE COMPLETE — row counts reconcile.")


if __name__ == "__main__":
    main()
