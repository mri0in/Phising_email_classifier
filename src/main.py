# src/main.py

"""
Application entry point for the phishing email classification system.

This module wires the required pipelines into the PipelineOrchestrator
and provides the command-line entry point for executing them.
"""

import logging

from src.orchestration.pipeline_orchestrator import PipelineOrchestrator
from src.pipelines.preprocessing_ppln import PreprocessingPipeline


def configure_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Initialize the application and execute the preprocessing pipeline."""

    configure_logging()

    preprocessing_pipeline = PreprocessingPipeline(
        input_path="data/raw/Phishing_Email.csv",
        output_path="data/processed/phishing_emails_processed.csv",
    )

    orchestrator = PipelineOrchestrator(
        pipelines={
            preprocessing_pipeline.name: preprocessing_pipeline,
        }
    )

    result = orchestrator.run("preprocessing")

    print("\nPipeline Result")
    print("----------------")
    print(f"Pipeline : {result.pipeline_name}")
    print(f"Success  : {result.success}")
    print(f"Message  : {result.message}")
    print(f"Metadata : {result.metadata}")


if __name__ == "__main__":
    main()