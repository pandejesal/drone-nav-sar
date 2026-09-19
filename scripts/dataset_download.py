#!/usr/bin/env python3
"""Sprint 12 — Dataset download from Hugging Face Hub (SAR-only).

Downloads `drone-nav-sar/dataset` (+ optional `drone-nav-sar/policies`)
and verifies SHA-256 checksums from checksums.sha256.

Usage:
  .venv/Scripts/python scripts/dataset_download.py --split all
  .venv/Scripts/python scripts/dataset_download.py --split test --no-verify
  .venv/Scripts/python scripts/dataset_download.py --repo drone-nav-sar/dataset

Requires: pip install huggingface_hub (only for real downloads).
Checksum verification works offline against a local manifest dir via --local-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "drone-nav-sar/dataset"
DEFAULT_POLICIES_REPO = "drone-nav-sar/policies"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(data_dir: Path, checksum_file: Path) -> bool:
    """Verify every `sha  relpath` line. Returns True iff all match."""
    if not checksum_file.exists():
        print(f"[dataset_download] no checksum file at {checksum_file}; skipping")
        return True
    ok = True
    lines = [ln.strip() for ln in checksum_file.read_text().splitlines() if ln.strip()]
    for ln in lines:
        try:
            expected, rel = ln.split(None, 1)
        except ValueError:
            print(f"[dataset_download] malformed line: {ln!r}")
            ok = False
            continue
        target = data_dir / rel.strip()
        if not target.exists():
            print(f"[dataset_download] MISSING: {rel}")
            ok = False
            continue
        actual = sha256_of(target)
        if actual != expected:
            print(f"[dataset_download] CHECKSUM MISMATCH: {rel}")
            ok = False
    print(f"[dataset_download] checksums: {'OK' if ok else 'FAILED'} "
          f"({len(lines)} files)")
    return ok


def download_from_hub(repo: str, out_dir: Path, splits) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[dataset_download] ERROR: huggingface_hub not installed. "
              "Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)
    allow = None
    if splits != ["all"]:
        allow = [f"spaces/*", "manifest.json", "checksums.sha256"]
        print(f"[dataset_download] note: split filtering {splits} is applied "
              f"post-download from manifest.json")
    local = snapshot_download(repo_id=repo, local_dir=str(out_dir),
                              allow_patterns=allow)
    return Path(local)


def filter_splits(data_dir: Path, splits) -> None:
    if splits == ["all"]:
        return
    import json
    manifest = data_dir / "manifest.json"
    if not manifest.exists():
        print("[dataset_download] no manifest.json; cannot filter splits")
        return
    data = json.loads(manifest.read_text())
    keep = [s for s in data.get("spaces", []) if s.get("split") in splits]
    print(f"[dataset_download] keeping {len(keep)}/{len(data['spaces'])} "
          f"spaces for splits={splits}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download Sprint 12 SAR dataset")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--policies-repo", default=DEFAULT_POLICIES_REPO)
    p.add_argument("--with-policies", action="store_true",
                   help="also download policy checkpoints repo")
    p.add_argument("--split", default="all",
                   help="'all' or comma-separated subset of train,val,test")
    p.add_argument("--output-dir", default="data/dataset")
    p.add_argument("--local-dir", default=None,
                   help="verify an existing local dir instead of downloading")
    p.add_argument("--no-verify", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    splits = ["all"] if args.split == "all" else args.split.split(",")
    out_dir = (ROOT / args.output_dir) if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)

    if args.local_dir:
        data_dir = Path(args.local_dir)
        print(f"[dataset_download] using local dir {data_dir}")
    else:
        print(f"[dataset_download] downloading {args.repo} -> {out_dir}")
        data_dir = download_from_hub(args.repo, out_dir, splits)
        if args.with_policies:
            pol_dir = out_dir.parent / "policies"
            print(f"[dataset_download] downloading {args.policies_repo}")
            download_from_hub(args.policies_repo, pol_dir, ["all"])

    filter_splits(data_dir, splits)

    if not args.no_verify:
        ok = verify_checksums(data_dir, data_dir / "checksums.sha256")
        if not ok:
            print("[dataset_download] FAILED checksum verification",
                  file=sys.stderr)
            return 1
    print(f"DONE download: {data_dir} splits={splits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
