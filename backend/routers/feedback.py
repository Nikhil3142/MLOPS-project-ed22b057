"""
feedback.py
─────────────────────────────────────────────────────────────────────────────
Feedback endpoint for ground truth logging.
Accepts actual charges to track real-world model performance decay.
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import logging
import os
from datetime import datetime
from fastapi import APIRouter
from backend.schemas.prediction import FeedbackRequest, FeedbackResponse
from backend.metrics import feedback_submissions_total

logger = logging.getLogger(__name__)
router = APIRouter()

FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "/app/data/feedback_log.csv")


@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(request: FeedbackRequest):
    """
    Accept ground truth labels for feedback loop logging.
    Logs actual vs predicted charges to a CSV file for
    later use in drift detection and retraining.
    """
    try:
        os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)

        file_exists = os.path.exists(FEEDBACK_LOG_PATH)
        with open(FEEDBACK_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "prediction_id", "actual_charge",
                "predicted_charge", "error", "input_features"
            ])
            if not file_exists:
                writer.writeheader()

            error = abs(request.actual_charge - request.predicted_charge)
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "prediction_id": request.prediction_id,
                "actual_charge": request.actual_charge,
                "predicted_charge": request.predicted_charge,
                "error": error,
                "input_features": str(request.input_features),
            })

        feedback_submissions_total.inc()
        logger.info(f"Feedback logged for prediction {request.prediction_id}")

        return FeedbackResponse(
            status="success",
            message="Feedback logged successfully"
        )

    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        return FeedbackResponse(
            status="error",
            message=str(e)
        )