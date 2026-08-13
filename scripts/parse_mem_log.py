#!/usr/bin/env python3
"""
parse_mem_log.py
=================
Reads a log of `docker stats --no-stream --format "{{.MemUsage}}"` lines
(one per poll, e.g. "45.2MiB / 1.952GiB") and prints the peak usage in MB.

Usage:
    python3 parse_mem_log.py log/mem_log.txt
"""
import re
import sys

UNIT_TO_MB = {
    "B": 1 / (1024 * 1024),
    "KIB": 1 / 1024,
    "KB": 1 / 1024,
    "MIB": 1.0,
    "MB": 1.0,
    "GIB": 1024.0,
    "GB": 1024.0,
}

LINE_RE = re.compile(r"([\d.]+)\s*([A-Za-z]+)\s*/")


def to_mb(value: float, unit: str) -> float:
    unit = unit.upper()
    if unit not in UNIT_TO_MB:
        raise ValueError(f"Unrecognized unit '{unit}' -- update UNIT_TO_MB if docker's format changed")
    return value * UNIT_TO_MB[unit]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mem_log.txt>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    peaks_mb = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.search(line)
            if not m:
                print(f"  [!] skipping unparseable line: {line!r}", file=sys.stderr)
                continue
            value, unit = m.groups()
            peaks_mb.append(to_mb(float(value), unit))

    if not peaks_mb:
        print("No parseable memory samples found.", file=sys.stderr)
        sys.exit(1)

    print(f"samples={len(peaks_mb)} idle_or_first={peaks_mb[0]:.1f}MB peak={max(peaks_mb):.1f}MB")


if __name__ == "__main__":
    main()
