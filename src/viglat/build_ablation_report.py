"""Backward-compatible shim for ``python -m viglat.build_ablation_report``."""

from viglat.cli.build_ablation_report import main

if __name__ == "__main__":
    raise SystemExit(main())
