"""Token verification using Gumbel-Max sampling for LLM outputs."""

from token_difr.verify import (
    SamplingMethod,
    TokenMetrics,
    TokenSequence,
    compute_metrics_summary,
    verify_outputs,
)

__version__ = "0.1.1"

__all__ = [
    "verify_outputs",
    "TokenSequence",
    "TokenMetrics",
    "SamplingMethod",
    "compute_metrics_summary",
    "__version__",
]
