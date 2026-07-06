"""Render a simple, dependency-free HTML dashboard of current opportunities.

Shows every market whose executable price cleared the model CI (the gate), flags
first, with the fields a human needs to decide: direction, model probability vs
market price (with the model CI), fee/slippage-adjusted EV, suggested size,
liquidity notes, and time-to-resolution. A prominent banner states how many are
actually flaggable and why (usually zero — nothing validated / no book).

Plain HTML on purpose: no Streamlit dependency, works offline, and can be served
by the existing ``serve.py`` or written to ``outputs/``.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime


def _fmt_ttr(event_time: datetime | None, ref: datetime) -> str:
    if event_time is None:
        return "?"
    secs = (event_time - ref).total_seconds()
    if secs < 0:
        return "resolved"
    d, rem = divmod(int(secs), 86400)
    h = rem // 3600
    return f"{d}d {h}h" if d else f"{h}h"


def _row_html(edge, event_time, ref) -> str:
    flag = edge.flaggable
    status = "FLAG" if flag else edge.reason
    ci = (f"[{edge.ci_lo:.2f},{edge.ci_hi:.2f}]"
          if edge.ci_lo is not None else "—")
    liq = "book" if edge.book_available else f"no-book (+{edge.slippage:.2f})"
    cells = [
        html.escape(edge.ticker),
        edge.side or "—",
        f"{edge.model_prob:.2f} {ci}" if edge.model_prob is not None else "—",
        f"{edge.market_price:.2f}" if edge.market_price is not None else "—",
        f"{edge.raw_edge:+.3f}" if edge.raw_edge is not None else "—",
        f"{edge.ev_per_contract:+.3f}" if edge.ev_per_contract is not None else "—",
        str(edge.contracts),
        html.escape(liq),
        _fmt_ttr(event_time, ref),
        f'<span class="{"flag" if flag else "muted"}">{html.escape(status)}</span>',
    ]
    cls = ' class="flagrow"' if flag else ""
    return "<tr" + cls + ">" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def render_dashboard(edges, signals, *, scan_ts: datetime | None = None,
                     validated_categories=None) -> str:
    ref = scan_ts or datetime.now(UTC)
    event_times = {s.ticker: s.event_time for s in signals}
    gated = [e for e in edges if e.gate_pass]
    gated.sort(key=lambda e: (not e.flaggable, -(e.ev_per_contract or -9)))
    n_flag = sum(e.flaggable for e in edges)

    rows = "\n".join(_row_html(e, event_times.get(e.ticker), ref) for e in gated) or (
        '<tr><td colspan="10" class="muted">No market cleared the model CI — all noise.</td></tr>'
    )
    banner_class = "ok" if n_flag else "warn"
    vc = validated_categories or []
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Kalshi Scanner — flags</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
 th {{ background: #f4f4f4; }}
 .flagrow {{ background: #eaf7ea; }}
 .flag {{ color: #167a16; font-weight: 600; }}
 .muted {{ color: #999; }}
 .banner {{ padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; }}
 .banner.ok {{ background: #eaf7ea; }}
 .banner.warn {{ background: #fff4e5; }}
</style></head><body>
<h1>Kalshi Opportunity Scanner</h1>
<div class="banner {banner_class}">
  <strong>{n_flag} flaggable</strong> of {len(edges)} evaluated ·
  validated categories: {html.escape(str(vc) if vc else "[] (nothing flaggable for real yet)")}
  · as of {html.escape(ref.isoformat())}
</div>
<p class="muted">Read-only. This tool never places orders. Rows below cleared the
model confidence-interval gate; only rows marked FLAG are actionable, and only in
validated categories.</p>
<table>
<tr><th>Ticker</th><th>Dir</th><th>Model (CI)</th><th>Mkt px</th><th>Edge</th>
    <th>EV/ctr</th><th>Size</th><th>Liquidity</th><th>Resolves</th><th>Status</th></tr>
{rows}
</table>
</body></html>"""


def write_dashboard(path, edges, signals, **kw) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = render_dashboard(edges, signals, **kw)
    out.write_text(content)
    return str(out)
