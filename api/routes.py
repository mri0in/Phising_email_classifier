# api/routes.py

"""
API routes for phishing email classification.

This module defines the HTTP endpoints exposed by the application.
It handles request validation and delegates machine-learning
inference to the existing inference pipeline.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.pipelines.inference_ppln import InferencePipeline


class APIRoutes:
    """
    Encapsulates the application's API routes.

    Supports JSON requests and raw plain-text email submissions
    without modifying the underlying inference pipeline.
    """

    def __init__(
        self,
        vectorizer_path: str | Path,
        model_path: str | Path,
    ) -> None:
        """
        Initialize API routes.

        Args:
            vectorizer_path: Path to the saved TF-IDF vectorizer.
            model_path: Path to the saved classification model.

        Raises:
            ValueError: If either path is empty.
        """

        if not vectorizer_path:
            raise ValueError("Vectorizer path cannot be empty.")

        if not model_path:
            raise ValueError("Model path cannot be empty.")

        self.logger = logging.getLogger(self.__class__.__name__)

        self.vectorizer_path = Path(vectorizer_path)
        self.model_path = Path(model_path)

        self.router = APIRouter()

        self._register_routes()

    def _register_routes(self) -> None:
        """Register all API endpoints."""

        self.router.add_api_route(
            "/",
            self.root,
            methods=["GET"],
        )

        self.router.add_api_route(
            "/health",
            self.health,
            methods=["GET"],
            response_model=HealthResponse,
        )

        self.router.add_api_route(
            "/predict",
            self.predict,
            methods=["POST"],
            response_model=PredictionResponse,
            openapi_extra={
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": PredictionRequest.model_json_schema(),
                            "example": {
                                "email_text": (
                                    "URGENT: Your account has been suspended. "
                                    "Click here to verify your account."
                                )
                            },
                        },
                        "text/plain": {
                            "schema": {
                                "type": "string",
                                "title": "Raw Email Text",
                            },
                            "example": (
                                "Subject: Payroll Verification\n\n"
                                "Dear Employee,\n\n"
                                "Please verify your payroll details."
                            ),
                        },
                    },
                }
            },
        )

    async def root(self) -> dict[str, str]:
        """Return basic API information."""

        return {
            "service": "Phishing Email Classifier API",
            "status": "running",
        }

    async def health(self) -> HealthResponse:
        """Return API and model artifact availability."""

        model_loaded = (
            self.vectorizer_path.is_file()
            and self.model_path.is_file()
        )

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
        )

    async def predict(
        self,
        http_request: Request,
    ) -> PredictionResponse:
        """
        Classify an email submitted as JSON or plain text.

        JSON requests use the PredictionRequest schema.
        Plain-text requests contain the raw email, including newlines.
        """

        self.logger.info("Received email classification request.")

        try:
            content_type = (
                http_request.headers.get("content-type", "")
                .split(";", maxsplit=1)[0]
                .strip()
                .lower()
            )

            if content_type == "application/json":
                try:
                    payload = await http_request.json()
                    request_data = PredictionRequest.model_validate(
                        payload
                    )
                except (
                    json.JSONDecodeError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Invalid JSON request. "
                            "Expected a non-empty email_text field."
                        ),
                    ) from exc

                email_text = request_data.email_text

            elif content_type == "text/plain":
                try:
                    email_text = (
                        await http_request.body()
                    ).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="Request body must contain valid UTF-8 text.",
                    ) from exc

            else:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Unsupported Content-Type. "
                        "Use application/json or text/plain."
                    ),
                )

            if not email_text.strip():
                raise HTTPException(
                    status_code=422,
                    detail="Email text cannot be empty.",
                )

            inference_pipeline = InferencePipeline(
                vectorizer_path=self.vectorizer_path,
                model_path=self.model_path,
                email_text=email_text,
            )

            result = inference_pipeline.run()

            if not result.success:
                raise RuntimeError(result.message)

            prediction = result.metadata

            self.logger.info(
                "Email classification completed successfully."
            )

            return PredictionResponse(
                prediction=prediction["prediction"],
                confidence=prediction["confidence"],
                decision_score=prediction["decision_score"],
                model_version=prediction["model_version"],
            )

        except HTTPException:
            raise

        except Exception as exc:
            self.logger.exception(
                "Email classification request failed."
            )

            raise HTTPException(
                status_code=500,
                detail="Email classification failed.",
            ) from exc