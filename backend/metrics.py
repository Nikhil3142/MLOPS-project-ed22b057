"""
metrics.py
─────────────────────────────────────────────────────────────────────────────
Prometheus custom metrics for FastAPI monitoring.
─────────────────────────────────────────────────────────────────────────────
"""

from prometheus_client import Counter, Histogram, Gauge

# ── Request metrics ───────────────────────────────────────────────────────────
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of failed predictions",
)

# ── Latency metrics ───────────────────────────────────────────────────────────
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
)

# ── Prediction value distribution ─────────────────────────────────────────────
predicted_cost_value = Histogram(
    "predicted_cost_value",
    "Distribution of predicted insurance cost values",
    buckets=[1000, 5000, 10000, 20000, 30000, 50000, 100000],
)

# ── Drift metrics ─────────────────────────────────────────────────────────────
feature_drift_score = Gauge(
    "feature_drift_score",
    "Per-feature drift score vs baseline",
    labelnames=["feature"],
)

# ── Feedback metrics ──────────────────────────────────────────────────────────
feedback_submissions_total = Counter(
    "feedback_submissions_total",
    "Total number of feedback submissions",
)