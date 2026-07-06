"""CLI for the Kalshi scanner.

    python -m kalshi_scanner scan-once     # one scan, then exit
    python -m kalshi_scanner run           # poll forever on the configured interval
    python -m kalshi_scanner status        # DB stats + coverage gaps

This is a READ-ONLY tool: it never places orders.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC

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
    ap.add_argument(
        "command",
        choices=["scan-once", "run", "status", "generate-signals", "evaluate-edges",
                 "flags", "paper-report", "monitor"],
    )
    ap.add_argument("--config", default=None, help="path to kalshi_scanner.yaml")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="stop 'run' after N scans (default: forever)")
    ap.add_argument("--scan-id", type=int, default=None,
                    help="scan to score for generate-signals (default: latest ok scan)")
    ap.add_argument("--signal-model", default="gbdt", help="research model to score with")
    ap.add_argument("--n-bootstrap", type=int, default=40, help="bootstrap replicas for the CI")
    ap.add_argument("--check-corpus", action="store_true",
                    help="monitor: also check transcript/GDELT staleness (builds the corpus)")
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

    if args.command == "generate-signals":
        from .signal_generator import build_default_signal_generator
        with SnapshotStore(config.db_path) as store:
            scan_id = args.scan_id or store.latest_ok_scan_id()
            if scan_id is None:
                print("no successful scan found; run scan-once first")
                return 1
            snapshots = store.load_snapshots(scan_id)
            print(f"scoring {len(snapshots)} snapshots from scan #{scan_id} "
                  f"(model={args.signal_model}, {args.n_bootstrap} bootstrap replicas)...")
            gen = build_default_signal_generator(
                config, store=store, model_name=args.signal_model, n_bootstrap=args.n_bootstrap
            )
            signals = gen.generate(snapshots, scan_id=scan_id)
            scored = [s for s in signals if s.is_scored]
            print(f"{len(scored)} scored / {len(signals)} total "
                  f"(validated categories: {config.validated_categories or '[]'})")
            for s in scored[:15]:
                tag = "VALIDATED" if s.validated else "unvalidated (not flaggable)"
                mkt = f"{s.market_implied_prob:.2f}" if s.market_implied_prob is not None else "n/a"
                print(f"  {s.ticker}: model={s.model_prob:.2f} "
                      f"[{s.ci_lo:.2f},{s.ci_hi:.2f}] mkt={mkt} — {tag}")
        return 0

    if args.command == "evaluate-edges":
        from .edge import EdgeEvaluator
        from .signal_generator import build_default_signal_generator
        with SnapshotStore(config.db_path) as store:
            scan_id = args.scan_id or store.latest_ok_scan_id()
            if scan_id is None:
                print("no successful scan found; run scan-once first")
                return 1
            snapshots = store.load_snapshots(scan_id)
            print(f"scoring + evaluating {len(snapshots)} markets from scan #{scan_id}...")
            gen = build_default_signal_generator(
                config, store=None, model_name=args.signal_model, n_bootstrap=args.n_bootstrap
            )
            signals = gen.generate(snapshots)
            edges = EdgeEvaluator(config.trading, kill_switch=config.kill_switch).evaluate_all(
                signals, scan_id=scan_id, store=store)
            flaggable = [e for e in edges if e.flaggable]
            print(f"{len(flaggable)} flaggable / {len(edges)} evaluated "
                  f"(validated categories: {config.validated_categories or '[]'})")
            # Show the closest-to-flaggable candidates (gate passed) for insight.
            gated = [e for e in edges if e.gate_pass]
            for e in gated[:15]:
                print(f"  {e.ticker} [{e.side}]: model={e.model_prob:.2f} "
                      f"px={e.market_price:.2f} ev/contract={e.ev_per_contract:+.3f} "
                      f"size={e.contracts} — {'FLAG' if e.flaggable else e.reason}")
            if not gated:
                print("  (no market's executable price is outside its model CI — all noise)")
        return 0

    if args.command == "flags":
        from datetime import datetime

        from mention_market.config import repo_root

        from .alerts import Alerter
        from .dashboard import write_dashboard
        from .edge import EdgeEvaluator
        from .flags import build_flags
        from .portfolio import PortfolioAllocator
        from .signal_generator import build_default_signal_generator
        with SnapshotStore(config.db_path) as store:
            scan_id = args.scan_id or store.latest_ok_scan_id()
            if scan_id is None:
                print("no successful scan found; run scan-once first")
                return 1
            snapshots = store.load_snapshots(scan_id)
            scan_ts = snapshots[0].scan_ts if snapshots else datetime.now(UTC)
            gen = build_default_signal_generator(
                config, store=None, model_name=args.signal_model, n_bootstrap=args.n_bootstrap
            )
            signals = gen.generate(snapshots)
            edges = EdgeEvaluator(config.trading, kill_switch=config.kill_switch).evaluate_all(
                signals, scan_id=scan_id, store=store)
            allocations = PortfolioAllocator(config.trading).allocate(edges)
            flags = build_flags(allocations, signals,
                                flagged_at=datetime.now(UTC), scan_id=scan_id)
            store.record_flags(flags)
            dash = write_dashboard(
                repo_root() / "outputs" / "kalshi_dashboard.html", edges, signals,
                scan_ts=scan_ts, validated_categories=config.validated_categories,
            )
            sent = Alerter(config.alerts).notify_flags(flags)
            print(f"{len(flags)} flag(s) logged, {len(sent)} alert(s) sent")
            print(f"dashboard -> {dash}")
        return 0

    if args.command == "paper-report":
        from .paper_trading import build_report
        with SnapshotStore(config.db_path) as store:
            positions = store.load_positions()
            r = build_report(positions)
            print(f"paper positions: {r.n_positions} ({r.n_settled} settled)")
            if r.n_settled:
                print(f"P&L:            ${r.pnl:,.2f}")
                print(f"hit rate:       {r.hit_rate:.1%}")
                print(f"realized Brier: {r.realized_brier:.4f}")
                print(f"edge realized:  {r.realized_edge_mean:+.4f} vs expected "
                      f"{r.expected_edge_mean:+.4f} (t={r.edge_diff_tstat:.2f}, "
                      f"p={r.edge_diff_pvalue:.3f})")
                print(f"max drawdown:   ${r.max_drawdown:,.2f}")
                print(f"toward-model:   {r.decay_toward_model:.1%} of flags")
                print(f"span:           {r.span_weeks:.1f} weeks")
            print(f"GO/NO-GO:       {'GO' if r.go else 'NO-GO'}")
            for note in r.notes:
                print(f"  - {note}")
        return 0

    if args.command == "monitor":
        from datetime import datetime

        from .data_quality import run_data_quality
        from .monitoring import assess_drift
        world = None
        if args.check_corpus:
            from .feature_resolver import default_research_world
            world = default_research_world()
        with SnapshotStore(config.db_path) as store:
            now = datetime.now(UTC)
            drift = assess_drift(store.load_positions(), config.monitoring)
            issues = run_data_quality(
                store, config.monitoring, now=now,
                expected_interval_s=config.poll_interval_s, world=world,
            )
            print(f"drift:  {'DEGRADED' if drift.degraded else 'ok'} — {drift.note}")
            print(f"kill_switch: {'ENGAGED (flagging paused)' if config.kill_switch else 'off'}")
            print(f"data-quality issues: {len(issues)}")
            for i in issues:
                print(f"  [{i.severity}] {i.check}: {i.detail}")
            if drift.degraded and config.alerts.enabled:
                from .alerts import Alerter
                Alerter(config.alerts)  # a real integration would push a drift alert here
        return 1 if drift.degraded or any(i.severity == "error" for i in issues) else 0

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
