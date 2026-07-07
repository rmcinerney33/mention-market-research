#!/usr/bin/env python
"""Serve the outputs directory (report, figures, CSVs) on localhost:8000.

Usage:
    python serve.py

Then visit http://localhost:8000 in your browser.
"""

import http.server
import socketserver
from pathlib import Path

PORT = 8000
OUTPUTS_DIR = Path(__file__).parent / "outputs"


class OutputsHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Serve from outputs/ directory
        if path == "/" or path == "":
            return str(OUTPUTS_DIR / "index.html")
        return str(OUTPUTS_DIR / path.lstrip("/"))


if not OUTPUTS_DIR.exists():
    print(f"Error: {OUTPUTS_DIR} does not exist. Run the pipeline first.")
    exit(1)

# Generate a simple index if it doesn't exist
index_path = OUTPUTS_DIR / "index.html"
if not index_path.exists():
    index_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mention-Market Research</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; max-width: 900px; }
        h1 { color: #333; }
        .section { margin: 30px 0; padding: 20px; background: #f5f5f5; border-radius: 8px; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .figure { margin: 20px 0; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Mention-Market Prediction & Market-Efficiency Research</h1>
    <p>All 7 phases implemented, end-to-end validated pipeline with synthetic ground truth.</p>

    <div class="section">
        <h2>📈 Kalshi Opportunity Scanner (dashboard)</h2>
        <p><a href="kalshi_dashboard.html">→ Open the live scanner dashboard</a></p>
        <p style="color:#666; font-size:14px;">Read-only. Never places orders. Currently 0 flaggable
        by design (no category validated, no order-book depth).</p>
    </div>

    <div class="section">
        <h2>📋 Main Report</h2>
        <p><a href="reports/REPORT.md">→ Read the full report (Markdown)</a></p>
    </div>

    <div class="section">
        <h2>📊 Figures</h2>
        <div class="figure">
            <h3>Calibration / Reliability Diagram</h3>
            <a href="figures/reliability.png"><img src="figures/reliability.png" style="max-width: 100%; height: auto;"></a>
        </div>
        <div class="figure">
            <h3>Brier Score by Model (95% Bootstrap CI)</h3>
            <a href="figures/brier_ci.png"><img src="figures/brier_ci.png" style="max-width: 100%; height: auto;"></a>
        </div>
        <div class="figure">
            <h3>AUC by Model (95% Bootstrap CI)</h3>
            <a href="figures/auc_ci.png"><img src="figures/auc_ci.png" style="max-width: 100%; height: auto;"></a>
        </div>
        <div class="figure">
            <h3>Mincer-Zarnowitz: Model vs Market Coefficients vs Lead Time</h3>
            <a href="figures/efficiency.png"><img src="figures/efficiency.png" style="max-width: 100%; height: auto;"></a>
        </div>
    </div>

    <div class="section">
        <h2>📈 Data Tables (CSV)</h2>
        <ul>
            <li><a href="reports/metrics.csv">metrics.csv</a> — Model comparison with bootstrap CIs</li>
            <li><a href="reports/dm_matrix.csv">dm_matrix.csv</a> — Diebold-Mariano pairwise p-values</li>
            <li><a href="reports/mz_results.csv">mz_results.csv</a> — Mincer-Zarnowitz regression coefficients</li>
            <li><a href="reports/oos_predictions.csv">oos_predictions.csv</a> — Out-of-sample predictions (all folds, all models)</li>
        </ul>
    </div>

    <div class="section">
        <h2>🚀 Getting Started</h2>
        <p>From the repo root:</p>
        <pre>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                              # 82 tests
python -m mention_market.pipeline   # regenerate outputs/
        </pre>
        <p><strong>Useful flags:</strong></p>
        <pre>
--no-bayesian           # skip the slow sampler for quick iterations
--market-efficiency 0.3 # inject a detectable model edge
--mz-method logit       # logistic regression instead of OLS
--events-per-pair 60    # smaller dataset for testing
        </pre>
    </div>

    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 14px;">
        <p>Locally served. <a href="file://""" + str(OUTPUTS_DIR) + """">View files on disk</a></p>
    </footer>
</body>
</html>
"""
    index_path.write_text(index_html)
    print(f"Generated {index_path}")

print(f"\n🚀 Serving {OUTPUTS_DIR} on http://localhost:{PORT}")
print("Press Ctrl+C to stop.\n")

with socketserver.TCPServer(("", PORT), OutputsHandler) as httpd:
    httpd.serve_forever()
