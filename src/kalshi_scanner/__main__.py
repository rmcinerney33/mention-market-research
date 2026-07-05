"""CLI for the Kalshi scanner.

    python -m kalshi_scanner scan-once     # one scan, then exit
    python -m kalshi_scanner run           # poll forever on the configured interval
    python -m kalshi_scanner status        # DB stats + coverage gaps

This is a READ-ONLY tool: it never places orders.
"""

from __future__ import annotations

import argparse
import logging

from .config import load_scanner_config
from .scanner import build_scanner
from .store import SnapshotStore


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kalshi_scanner", description=__doc__)
    ap.add_argument("command", choices=["scan-once", "run", "status"])
    ap.add_argument("--config", default=None, help="path to kalshi_scanner.yaml")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="stop 'run' after N scans (default: forever)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    config = load_scanner_config(args.config)

    if args.command == "status":
        with SnapshotStore(config.db_path) as store:
            latest = store.latest_run()
            print(f"db:               {config.db_path}")
            print(f"scan runs:        {store.count_runs()} "
                  f"(ok={store.count_runs('ok')}, failed={store.count_runs('failed')})")
            print(f"snapshots stored: {store.count_snapshots()}")
            if latest is not None:
                print(f"latest run:       #{latest['scan_id']} {latest['status']} "
                      f"@ {latest['started_at']} (n_markets={latest['n_markets']})")
            gaps = store.find_gaps(config.poll_interval_s)
            print(f"coverage gaps:    {len(gaps)}")
            for g in gaps[:10]:
                print(f"  - {g}")
        return 0

    scanner = build_scanner(config)
    if args.command == "scan-once":
        result = scanner.scan_once()
        status = "ok" if result.ok else f"FAILED ({result.error})"
        print(f"scan {status}: {result.n_markets} markets matched "
              f"validated={config.validated_categories or '[]'}")
        return 0 if result.ok else 1

    if args.command == "run":
        print(f"scanning every {config.poll_interval_s}s "
              f"(Ctrl+C to stop). Categories: {[c.name for c in config.categories]}")
        try:
            scanner.run_forever(max_iterations=args.max_iterations)
        except KeyboardInterrupt:
            print("\nstopped.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
