# src/main.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from src.orchestration.best_model_selection import BestModelSelection
from src.orchestration.model_comparison import ModelComparisonRunner
from src.orchestration.pipeline_orchestrator import PipelineOrchestrator
from src.pipelines.evaluation_ppln import EvaluationPipeline
from src.pipelines.feature_ppln import FeaturePipeline
from src.pipelines.inference_ppln import InferencePipeline
from src.pipelines.preprocessing_ppln import PreprocessingPipeline
from src.pipelines.training_ppln import TrainingPipeline


logger = logging.getLogger(__name__)


SUPPORTED_PIPELINES = (
    "preprocessing",
    "features",
    "training",
    "inference",
    "evaluation",
)

SUPPORTED_MODELS = (
    "logistic_regression",
    "linear_svm",
    "naive_bayes",
    "random_forest",
)


def configure_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load application configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration is not a mapping.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Configuration path must be a file: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Phishing Email Detection ML pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--pipeline",
        choices=SUPPORTED_PIPELINES,
        help="Pipeline to execute.",
    )

    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        help="Model to use when running the training pipeline.",
    )

    parser.add_argument(
        "--compare-models",
        action="store_true",
        help=(
            "Train and evaluate all supported models "
            "for model comparison."
        ),
    )

    return parser.parse_args()


def build_pipelines(
    config: dict[str, Any],
    model_name: str,
    inference_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build all individual application pipelines.

    Args:
        config: Application configuration.
        model_name: Model used by the training pipeline.
        inference_model_path: Optional model artifact path to use
            specifically for inference.

    Returns:
        Dictionary containing configured pipeline instances.
    """

    paths = config["paths"]
    random_state = config["application"]["random_state"]

    preprocessing_pipeline = PreprocessingPipeline(
        input_path=paths["raw_data"],
        output_path=paths["processed_data"],
    )

    feature_pipeline = FeaturePipeline(
    input_path=paths["processed_data"],
    output_dir=paths["features_dir"],
    vectorizer_path=paths["vectorizer"],
    random_state=random_state,
    tfidf_config=config["features"]["tfidf"],
    split_config=config["features"]["split"],
    )

    training_pipeline = TrainingPipeline(
        features_dir=paths["features_dir"],
        model_output_path=paths["selected_model"],
        random_state=random_state,
        model_name=model_name,
        model_parameters=config["training"]["models"][model_name],
        mlflow_config=config["mlflow"],
    )

    inference_model = (
        inference_model_path
        if inference_model_path is not None
        else paths["selected_model"]
    )

    inference_pipeline = InferencePipeline(
        vectorizer_path=paths["vectorizer"],
        model_path=inference_model,
        email_text=config["inference"]["email_text"],
    )

    evaluation_pipeline = EvaluationPipeline(
        features_dir=paths["features_dir"],
        model_path=paths["selected_model"],
        metrics_output_path=paths["evaluation_metrics"],
    )

    return {
        preprocessing_pipeline.name: preprocessing_pipeline,
        feature_pipeline.name: feature_pipeline,
        training_pipeline.name: training_pipeline,
        inference_pipeline.name: inference_pipeline,
        evaluation_pipeline.name: evaluation_pipeline,
    }


def run_pipeline(
    config: dict[str, Any],
    pipeline_name: str,
    model_name: str,
    inference_model_path: str | Path | None = None,
) -> None:
    """Run one individual application pipeline.

    Args:
        config: Application configuration.
        pipeline_name: Name of the pipeline to execute.
        model_name: Model used by the training pipeline.
        inference_model_path: Optional model artifact path for
            inference.
    """

    pipelines = build_pipelines(
        config=config,
        model_name=model_name,
        inference_model_path=inference_model_path,
    )

    orchestrator = PipelineOrchestrator(
        pipelines=pipelines,
    )

    result = orchestrator.run(pipeline_name)

    print("\nPipeline Result")
    print("----------------")
    print(f"Pipeline : {result.pipeline_name}")
    print(f"Success  : {result.success}")
    print(f"Message  : {result.message}")
    print(f"Metadata : {result.metadata}")


def run_model_comparison(
    config: dict[str, Any],
) -> None:
    """Run the complete model comparison workflow.

    The comparison workflow independently trains all supported
    classifiers and persists their validation metrics to a CSV file.

    Model selection and inference are intentionally not performed
    inside this workflow.

    Args:
        config: Application configuration.
    """

    paths = config["paths"]
    random_state = config["application"]["random_state"]
    model_configs = config["training"]["models"]

    comparison_runner = ModelComparisonRunner(
        features_dir=paths["features_dir"],
        output_dir=paths["model_comparison"],
        metrics_output_path=paths["model_comparison_metrics"],
        random_state=random_state,
        model_configs=model_configs,
    )

    comparison_results = comparison_runner.run()

    print("\nModel Comparison Result")
    print("-----------------------")

    for model_name, metadata in comparison_results.items():
        print(f"\nModel: {model_name}")
        print(f"Metadata: {metadata}")

    logger.info(
        "Model comparison completed successfully."
    )

    logger.info(
        "Comparison results saved to: %s",
        paths["model_comparison_metrics"],
    )

def select_model_for_inference(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Select a model using the persisted comparison results.

    The comparison CSV acts as the persisted handoff between
    model comparison and inference.

    Args:
        config: Application configuration.

    Returns:
        Metadata describing the selected model and its artifact.
    """

    paths = config["paths"]

    best_model_selection = BestModelSelection(
        selection_config=config["model_selection"],
        comparison_results_path=paths["model_comparison_metrics"],
    )

    selected_model = best_model_selection.select()

    logger.info(
        "Model selected for inference: %s",
        selected_model["model"],
    )

    logger.info(
        "Selected model metric: %s = %.6f",
        selected_model["metric"],
        selected_model["score"],
    )

    logger.info(
        "Selected model artifact: %s",
        selected_model["model_artifact"],
    )

    return selected_model


def run_inference(
    config: dict[str, Any],
    model_name: str,
) -> None:
    """Run inference using the model selected from comparison results.

    The selection policy comes from YAML, while the candidate metrics
    come from the already-produced comparison CSV.

    Args:
        config: Application configuration.
        model_name: Default model name used when constructing the
            training pipeline object.
    """

    selected_model = select_model_for_inference(config)

    selected_model_artifact = selected_model["model_artifact"]

    run_pipeline(
        config=config,
        pipeline_name="inference",
        model_name=model_name,
        inference_model_path=selected_model_artifact,
    )


def validate_configuration(
    config: dict[str, Any],
) -> None:
    """Validate required application configuration.

    Args:
        config: Application configuration.

    Raises:
        ValueError: If required configuration sections or values
            are missing or invalid.
    """

    required_sections = (
        "application",
        "paths",
        "inference",
        "model_selection",
    )

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise ValueError(
            "Missing required configuration section(s): "
            f"{missing_sections}"
        )

    application_config = config["application"]

    if not isinstance(application_config, dict):
        raise ValueError(
            "'application' configuration must be a mapping."
        )

    default_model = application_config.get("default_model")

    if default_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid default model '{default_model}'. "
            f"Supported models: {SUPPORTED_MODELS}"
        )

    default_pipeline = application_config.get("default_pipeline")

    if default_pipeline not in SUPPORTED_PIPELINES:
        raise ValueError(
            f"Invalid default pipeline '{default_pipeline}'. "
            f"Supported pipelines: {SUPPORTED_PIPELINES}"
        )


def main() -> None:
    """Application entry point."""

    configure_logging()

    arguments = parse_arguments()

    config = load_config(arguments.config)

    validate_configuration(config)

    if arguments.compare_models and arguments.pipeline:
        raise ValueError(
            "--compare-models cannot be combined with --pipeline."
        )

    if arguments.compare_models and arguments.model:
        raise ValueError(
            "--compare-models cannot be combined with --model."
        )

    if arguments.model and arguments.pipeline != "training":
        raise ValueError(
            "--model can only be used with the training pipeline."
        )

    # ---------------------------------------------------------------
    # Workflow 1: Model comparison
    #
    # Train all candidate models, evaluate them, write the comparison
    # CSV, and stop.
    # ---------------------------------------------------------------

    if arguments.compare_models:
        logger.info(
            "Starting model comparison workflow."
        )

        run_model_comparison(config)

        return

    pipeline_name = (
        arguments.pipeline
        or config["application"]["default_pipeline"]
    )

    model_name = (
        arguments.model
        or config["application"]["default_model"]
    )

    # ---------------------------------------------------------------
    # Workflow 2: Automatic best-model inference
    #
    # Read the completed comparison CSV, apply the YAML selection
    # policy, obtain the selected model artifact, and run inference.
    # ---------------------------------------------------------------

    if pipeline_name == "inference":
        logger.info(
            "Starting automatic best-model inference workflow."
        )

        run_inference(
            config=config,
            model_name=model_name,
        )

        return

    # ---------------------------------------------------------------
    # Workflow 3: Normal individual pipeline execution
    # ---------------------------------------------------------------

    logger.info(
        "Starting pipeline '%s' with model '%s'.",
        pipeline_name,
        model_name,
    )

    run_pipeline(
        config=config,
        pipeline_name=pipeline_name,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()