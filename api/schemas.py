# api/schemas.py

"""
Pydantic schemas for the phishing email classification API.

This module defines the request and response contracts used by the API.
It contains no machine-learning or inference logic.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request payload for email classification.
    """

    email_text: str = Field(
        ...,
        min_length=1,
        description="Email text to classify.",
        examples=[
            "URGENT: Your account has been suspended. "
            "Click here immediately to verify your account."
        ],
    )


class PredictionResponse(BaseModel):
    """Response schema for phishing email classification."""

    prediction: str = Field(
        ...,
        description="Predicted email class: phishing or safe.",
        examples=["phishing"],
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated prediction probability, if available. "
            "Currently null because the Linear SVM is not calibrated."
        ),
        examples=[None],
    )

    decision_score: float | None = Field(
        default=None,
        description=(
            "Raw SVM decision margin. This is not a probability "
            "and should not be interpreted as confidence."
        ),
        examples=[1.5113],
    )

    model_version: str = Field(
        ...,
        description="Version identifier of the model used for inference.",
        examples=["v1"],
    )


class HealthResponse(BaseModel):
    """
    Response returned by the API health endpoint.
    """

    status: str = Field(
        ...,
        description="Current API service status.",
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the inference model is available.",
    )