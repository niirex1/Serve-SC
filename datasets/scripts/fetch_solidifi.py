#!/usr/bin/env python3
"""Fetch the SolidiFI-benchmark dataset from its canonical repository.

    python scripts/fetch_solidifi.py [--dest solidifi-benchmark]

SolidiFI is NOT bundled in this package: at the time of writing its repository
did not advertise a redistribution licence, and re-hosting a dataset without a
clear licence is not safe. This script downloads it from the source so you get
the authoritative version.

    >>> VERIFY SolidiFI's LICENSE before adding it to a public repository. <<<

SolidiFI injects known bug types into seed contracts. After fetching, adapt
convert_smartbugs_to_ir.py's mapping logic to SolidiFI's directory/label layout
to produce IR records (see DATASETS.md).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "DependableSystemsLab/SolidiFI-benchmark"
BRANCHES = ("master", "main")


def _try_zip(dest: Path) -> bool:
    for br in BRANCHES:
        url = f"https://codeload.github.com/{REPO}/zip/refs/heads/{br}"
        try:
            print(f"trying {url}")
            tmp = dest.with_suffix(".zip")
            urllib.request.urlretrieve(url, tmp)
            with zipfile.ZipFile(tmp) as zf:
                zf.extractall(dest.parent)
            tmp.unlink(missing_ok=True)
            print(f"extracted under {dest.parent}/")
            return True
        except Exception as ex:  # noqa
            print(f"  failed ({ex})")
    return False


def _try_git(dest: Path) -> bool:
    try:
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/{REPO}.git", str(dest)], check=True)
        return True
    except Exception as ex:  # noqa
        print(f"git clone failed ({ex})")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="solidifi-benchmark")
    args = ap.parse_args()
    dest = Path(args.dest)

    print("=" * 72)
    print("SolidiFI is fetched from its source and NOT redistributed by this "
          "package.\nVERIFY ITS LICENSE before adding it to any public repo.")
    print("=" * 72)

    if _try_zip(dest) or _try_git(dest):
        print("done.")
    else:
        print(f"\nCould not fetch automatically. Clone manually:\n"
              f"  git clone https://github.com/{REPO}.git {dest}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
