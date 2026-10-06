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
    """
    Response returned after classifying an email.
    """

    prediction: str = Field(
        ...,
        description="Classification result: safe or phishing.",
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Model probability for the predicted class, when supported "
            "by the classifier."
        ),
    )

    decision_score: float | None = Field(
        default=None,
        description=(
            "Raw decision score returned by classifiers that do not "
            "provide probability estimates."
        ),
    )

    model_version: str = Field(
        ...,
        description="Version identifier of the deployed model.",
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