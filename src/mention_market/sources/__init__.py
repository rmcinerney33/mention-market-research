"""Ingestion adapters for external data sources.

Each adapter is a thin, well-documented client that returns pipeline schema
objects (``schema.py``) or plain DataFrames. They hit real endpoints; none are
required for the end-to-end synthetic demonstration, so the analysis stack can
be developed and tested without credentials or historical-data access.

Known access constraints are documented per-adapter and summarized in the
project README.
"""
