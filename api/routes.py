# api/routes.py

"""
API routes for phishing email classification.

This module defines the HTTP endpoints exposed by the application.
It contains API-layer logic only and delegates machine-learning
inference to the existing inference pipeline.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.pipelines.inference_ppln import InferencePipeline


class APIRoutes:
    """
    Encapsulates the application's API routes.

    The class is responsible for connecting HTTP requests to the
    existing inference pipeline without implementing ML logic itself.
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
        )

    async def root(self) -> dict[str, str]:
        """
        Return basic API information.
        """

        return {
            "service": "Phishing Email Classifier API",
            "status": "running",
        }

    async def health(self) -> HealthResponse:
        """
        Return API and model health status.
        """

        model_loaded = (
            self.vectorizer_path.exists()
            and self.vectorizer_path.is_file()
            and self.model_path.exists()
            and self.model_path.is_file()
        )

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
        )

    async def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """
        Classify an email using the existing inference pipeline.

        Args:
            request: Validated prediction request.

        Returns:
            PredictionResponse containing the prediction,
            confidence, and model version.

        Raises:
            HTTPException: If inference fails.
        """

        self.logger.info("Received email classification request.")

        try:
            inference_pipeline = InferencePipeline(
                vectorizer_path=self.vectorizer_path,
                model_path=self.model_path,
                email_text=request.email_text,
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
                model_version=prediction["model_version"],
            )

        except Exception as exc:
            self.logger.exception(
                "Email classification request failed."
            )

            raise HTTPException(
                status_code=500,
                detail="Email classification failed.",
            ) from exc