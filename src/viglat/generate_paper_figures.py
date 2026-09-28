"""Backward-compatible shim for ``python -m viglat.generate_paper_figures``."""

from viglat.cli.generate_paper_figures import main

if __name__ == "__main__":
    raise SystemExit(main())
