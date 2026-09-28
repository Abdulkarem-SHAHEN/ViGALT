"""Backward-compatible shim for ``python -m viglat.refresh_relevancy_metrics``."""

from viglat.cli.refresh_relevancy_metrics import main

if __name__ == "__main__":
    raise SystemExit(main())
