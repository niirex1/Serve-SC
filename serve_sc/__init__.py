"""SERVE-SC reference implementation.

Two-stage smart-contract risk prioritisation: HGT-based multi-label detection
(Stage 1), a parameter-free exploitability gate, and an L2-regularised logistic
service-impact model (Stage 2). See README.md and DATA_SCHEMA.md.
"""
from . import config, metrics
from .detect import Detector
from .exploitability import exploitability, precondition_satisfaction
from .graph import build_graph
from .pipeline import run_pipeline, write_results
from .risk import RiskModel

__all__ = [
    "config", "metrics", "Detector", "exploitability",
    "precondition_satisfaction", "build_graph", "run_pipeline",
    "write_results", "RiskModel",
]
__version__ = "0.1.0"
