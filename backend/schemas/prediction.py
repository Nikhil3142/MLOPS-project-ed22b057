"""
prediction.py
─────────────────────────────────────────────────────────────────────────────
Pydantic schemas for request and response validation.
─────────────────────────────────────────────────────────────────────────────
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional


class PredictionRequest(BaseModel):
    """Input schema for /predict endpoint."""

    age: int = Field(..., ge=18, le=100, description="Age of the individual")
    sex: str = Field(..., description="Sex of the individual: male or female")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    children: int = Field(..., ge=0, le=10, description="Number of dependents")
    smoker: str = Field(..., description="Smoker status: yes or no")
    region: str = Field(..., description="Region: northeast, northwest, southeast, southwest")

    @validator("sex")
    def validate_sex(cls, v):
        if v.lower() not in {"male", "female"}:
            raise ValueError("sex must be 'male' or 'female'")
        return v.lower()

    @validator("smoker")
    def validate_smoker(cls, v):
        if v.lower() not in {"yes", "no"}:
            raise ValueError("smoker must be 'yes' or 'no'")
        return v.lower()

    @validator("region")
    def validate_region(cls, v):
        valid = {"northeast", "northwest", "southeast", "southwest"}
        if v.lower() not in valid:
            raise ValueError(f"region must be one of {valid}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "sex": "male",
                "bmi": 28.5,
                "children": 2,
                "smoker": "no",
                "region": "northeast"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for /predict endpoint."""

    predicted_charge: float = Field(..., description="Predicted annual insurance charge in USD")
    feature_importance: Dict[str, float] = Field(..., description="SHAP feature importance values")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class FeedbackRequest(BaseModel):
    """Input schema for /feedback endpoint."""

    prediction_id: str = Field(..., description="ID of the prediction to provide feedback for")
    actual_charge: float = Field(..., ge=0, description="Actual insurance charge")
    predicted_charge: float = Field(..., ge=0, description="Predicted insurance charge")
    input_features: Dict = Field(..., description="Original input features")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "abc123",
                "actual_charge": 12000.0,
                "predicted_charge": 11500.0,
                "input_features": {
                    "age": 35,
                    "sex": "male",
                    "bmi": 28.5,
                    "children": 2,
                    "smoker": "no",
                    "region": "northeast"
                }
            }
        }


class FeedbackResponse(BaseModel):
    """Output schema for /feedback endpoint."""

    status: str
    message: str


class HealthResponse(BaseModel):
    """Output schema for /health endpoint."""

    status: str
    service: str


class ReadyResponse(BaseModel):
    """Output schema for /ready endpoint."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Output schema for /model-info endpoint."""

    model_name: str
    model_version: str
    model_stage: str
    rmse: Optional[float] = None