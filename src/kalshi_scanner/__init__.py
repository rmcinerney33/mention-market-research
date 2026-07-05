"""Kalshi Opportunity Scanner.

A **read-only** scanner and alerting tool that sits on top of the
``mention_market`` research package. It polls live Kalshi markets, will (in
later phases) run the research repo's validated models against them, and flags
opportunities for a human to review and execute *manually*.

Hard rules (enforced throughout, see README):

- This package never places orders. There is no execution code, not even
  disabled stubs.
- Real-money use is gated behind a paper-trading period.
- Only categories the research repo has *statistically validated* may ever be
  flagged. If nothing is validated, the system flags nothing — by design.

Phase 1 (this commit) implements only the market scanner + snapshot store.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
