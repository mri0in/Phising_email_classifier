# api/main.py

"""
FastAPI application entry point.

This module creates and configures the FastAPI application.
Machine-learning logic remains inside the existing src package.
"""

import logging

from fastapi import FastAPI

from api.routes import APIRoutes


def configure_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """

    configure_logging()

    app = FastAPI(
        title="Phishing Email Classifier API",
        description=(
            "API for classifying emails as safe or phishing "
            "using a trained machine-learning model."
        ),
        version="1.0.0",
    )

    api_routes = APIRoutes(
        vectorizer_path="artifacts/models/tfidf_vectorizer.joblib",
        model_path="artifacts/models/phishing_classifier.joblib",
    )

    app.include_router(api_routes.router)

    return app


app = create_app()