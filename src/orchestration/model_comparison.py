# src/orchestration/model_comparison.py

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from src.orchestration.pipeline_orchestrator import PipelineOrchestrator
from src.pipelines.training_ppln import TrainingPipeline


class ModelComparisonRunner:
    """Coordinate training, evaluation, and comparison of multiple models.

    The comparison runner creates a separate TrainingPipeline instance for
    each configured classifier and executes that pipeline through the
    PipelineOrchestrator.

    The runner is responsible for:
        - coordinating model training runs
        - collecting evaluation metadata
        - saving comparison model artifacts
        - persisting consolidated comparison metrics

    The actual model training and evaluation remain inside TrainingPipeline.
    """

    DEFAULT_MODELS = (
        "logistic_regression",
        "linear_svm",
        "naive_bayes",
        "random_forest",
    )

    COMPARISON_COLUMNS = (
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "training_rows",
        "validation_rows",
        "feature_count",
        "model_artifact",
        "mlflow_run_id",
    )

    def __init__(
        self,
        features_dir: str | Path,
        output_dir: str | Path,
        metrics_output_path: str | Path,
        random_state: int = 42,
        models: tuple[str, ...] | None = None,
        model_configs: dict[str, dict[str, Any]] | None = None,
        mlflow_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the model comparison runner.

        Args:
            features_dir: Directory containing generated feature artifacts.
            output_dir: Directory where comparison model artifacts are stored.
            metrics_output_path: Path where consolidated comparison metrics
                will be saved as a CSV file.
            random_state: Random seed used for reproducible training.
            models: Optional tuple containing model names to compare.
                If omitted, all supported models are compared.
            model_configs: Mapping containing model-specific training
                parameters loaded from external configuration.
            mlflow_config: Mapping containing MLflow tracking configuration.

        Raises:
            ValueError: If no models are configured.
            TypeError: If model_configs has an invalid type.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.metrics_output_path = Path(metrics_output_path)
        self.random_state = random_state
        self.models = models or self.DEFAULT_MODELS
        self.mlflow_config = mlflow_config or {}

        if not self.models:
            raise ValueError(
                "At least one model must be provided for comparison."
            )

        if model_configs is not None and not isinstance(
            model_configs,
            dict,
        ):
            raise TypeError(
                "model_configs must be a dictionary."
            )

        self.model_configs = model_configs or {}

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metrics_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self) -> dict[str, dict[str, Any]]:
        """Run training and evaluation for every configured model.

        Returns:
            Dictionary mapping each model name to its pipeline metadata.

        Raises:
            FileNotFoundError: If the feature directory does not exist.
            ValueError: If an unsupported model is configured or a model
                configuration is missing.
            RuntimeError: If model training or comparison persistence fails.
        """

        self._validate_inputs()

        comparison_results: dict[str, dict[str, Any]] = {}

        self.logger.info(
            "Starting model comparison for %d models.",
            len(self.models),
        )

        for model_name in self.models:
            self.logger.info(
                "Starting comparison run for model: %s",
                model_name,
            )

            training_pipeline = self._create_training_pipeline(
                model_name=model_name,
            )

            orchestrator = PipelineOrchestrator(
                pipelines={
                    training_pipeline.name: training_pipeline,
                }
            )

            try:
                result = orchestrator.run("training")
            except Exception as exc:
                self.logger.exception(
                    "Model comparison failed for model: %s",
                    model_name,
                )
                raise RuntimeError(
                    f"Model comparison failed for '{model_name}'."
                ) from exc

            if not result.success:
                raise RuntimeError(
                    f"Training pipeline failed for '{model_name}': "
                    f"{result.message}"
                )

            comparison_results[model_name] = result.metadata

            self.logger.info(
                "Comparison run completed successfully for model: %s",
                model_name,
            )

        self._save_comparison_results(
            comparison_results=comparison_results,
        )

        self.logger.info(
            "Model comparison completed successfully for all models."
        )

        return comparison_results

    def _create_training_pipeline(
        self,
        model_name: str,
    ) -> TrainingPipeline:
        """Create a TrainingPipeline configured for one classifier.

        Args:
            model_name: Name of the classifier to train.

        Returns:
            Configured TrainingPipeline instance.

        Raises:
            ValueError: If model-specific configuration is missing.
        """

        if model_name not in self.model_configs:
            raise ValueError(
                f"Training configuration is missing for model: "
                f"'{model_name}'."
            )

        model_output_path = self._build_model_output_path(
            model_name=model_name,
        )

        return TrainingPipeline(
            features_dir=self.features_dir,
            model_output_path=model_output_path,
            random_state=self.random_state,
            model_name=model_name,
            model_parameters=self.model_configs[model_name],
            mlflow_config=self.mlflow_config,
        )

    def _build_model_output_path(
        self,
        model_name: str,
    ) -> Path:
        """Build the model artifact path for a comparison run.

        Args:
            model_name: Name of the classifier.

        Returns:
            Path for the serialized model artifact.
        """

        return self.output_dir / (
            f"phishing_classifier_{model_name}.joblib"
        )

    def _save_comparison_results(
        self,
        comparison_results: dict[str, dict[str, Any]],
    ) -> None:
        """Save consolidated model comparison metrics to CSV.

        Args:
            comparison_results: Metadata returned by each training pipeline.

        Raises:
            RuntimeError: If the CSV cannot be written.
        """

        self.logger.info(
            "Saving model comparison results to: %s",
            self.metrics_output_path,
        )

        try:
            with self.metrics_output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=self.COMPARISON_COLUMNS,
                )

                writer.writeheader()

                for model_name, metadata in comparison_results.items():
                    writer.writerow(
                        self._build_comparison_row(
                            model_name=model_name,
                            metadata=metadata,
                        )
                    )

        except OSError as exc:
            self.logger.exception(
                "Failed to save model comparison results."
            )
            raise RuntimeError(
                "Failed to save model comparison results."
            ) from exc

        self.logger.info(
            "Model comparison results saved successfully."
        )

    def _build_comparison_row(
        self,
        model_name: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one CSV row from training pipeline metadata.

        Args:
            model_name: Name of the classifier.
            metadata: Metadata returned by TrainingPipeline.

        Returns:
            Dictionary representing one comparison result row.
        """

        metrics = metadata.get("validation_metrics", {})

        return {
            "model": model_name,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1_score"),
            "training_rows": metadata.get("training_rows"),
            "validation_rows": metadata.get("validation_rows"),
            "feature_count": metadata.get("feature_count"),
            "mlflow_run_id": metadata.get("mlflow_run_id"),
            "model_artifact": str(
                self._build_model_output_path(
                    model_name=model_name,
                )
            ),
        }

    def _validate_inputs(self) -> None:
        """Validate model comparison configuration.

        Raises:
            FileNotFoundError: If the feature directory does not exist.
            ValueError: If the feature path is invalid, a model is
                unsupported, or model configuration is missing.
        """

        if not self.features_dir.exists():
            raise FileNotFoundError(
                f"Feature directory does not exist: {self.features_dir}"
            )

        if not self.features_dir.is_dir():
            raise ValueError(
                f"Feature path must be a directory: {self.features_dir}"
            )

        supported_models = set(self.DEFAULT_MODELS)

        invalid_models = set(self.models) - supported_models

        if invalid_models:
            raise ValueError(
                f"Unsupported model(s): {sorted(invalid_models)}. "
                f"Supported models: {sorted(supported_models)}"
            )

        missing_configs = [
            model_name
            for model_name in self.models
            if model_name not in self.model_configs
        ]

        if missing_configs:
            raise ValueError(
                "Training configuration is missing for model(s): "
                f"{missing_configs}"
            )