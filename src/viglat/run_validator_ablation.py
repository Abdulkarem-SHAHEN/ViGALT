"""Backward-compatible shim for ``python -m viglat.run_validator_ablation``."""

from viglat.cli.run_validator_ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
