# src/tracking/mlflow_tracker.py

"""
MLflow experiment tracking utilities for the phishing email classifier.

This module provides a small abstraction around MLflow so that the
training and orchestration layers do not depend directly on MLflow APIs.

The tracker is responsible for:
    1. Configuring the MLflow tracking URI.
    2. Managing MLflow experiments.
    3. Starting and ending MLflow runs.
    4. Logging model parameters.
    5. Logging evaluation metrics.
    6. Logging persisted artifacts.

Model training and evaluation logic remain outside this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow


class MLflowTracker:
    """
    Manage MLflow experiment tracking for model training runs.

    This class acts as an infrastructure boundary between the ML
    pipeline and MLflow. TrainingPipeline should provide tracking
    information to this class rather than interacting with MLflow
    directly.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        """
        Initialize the MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: Optional MLflow tracking server URI.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If experiment_name is empty.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if not isinstance(experiment_name, str):
            raise TypeError(
                "experiment_name must be a string."
            )

        if not experiment_name.strip():
            raise ValueError(
                "experiment_name must not be empty."
            )

        if tracking_uri is not None and not isinstance(
            tracking_uri,
            str,
        ):
            raise TypeError(
                "tracking_uri must be a string or None."
            )

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        self.logger.info(
            "MLflow tracker initialized | experiment=%s | tracking_uri=%s",
            self.experiment_name,
            mlflow.get_tracking_uri(),
        )

    def start_run(self, run_name: str | None = None) -> Any:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional human-readable name for the run.

        Returns:
            The active MLflow Run object.
        """

        self.logger.info(
            "Starting MLflow run | run_name=%s",
            run_name,
        )

        active_run = mlflow.start_run(
            run_name=run_name
        )

        self.logger.info(
            "MLflow run started | run_id=%s | run_name=%s",
            active_run.info.run_id,
            run_name,
        )

        return active_run
    
    def log_parameters(
        self,
        parameters: dict[str, Any],
    ) -> None:
        """
        Log model or pipeline parameters to the active MLflow run.

        Values are converted to strings when necessary because MLflow
        parameters must be serializable as parameter values.

        Args:
            parameters: Mapping of parameter names to parameter values.

        Raises:
            TypeError: If parameters is not a dictionary.
        """

        if not isinstance(parameters, dict):
            raise TypeError(
                "parameters must be a dictionary."
            )

        normalized_parameters = {
            str(key): self._normalize_parameter_value(value)
            for key, value in parameters.items()
        }

        mlflow.log_params(normalized_parameters)

        self.logger.info(
            "Logged %d parameters to MLflow.",
            len(normalized_parameters),
        )


    def set_tags(self, tags: dict[str, Any]) -> None:
        """
        Set descriptive tags for the active MLflow run.

        Args:
            tags: Mapping of tag names to tag values.

        Raises:
            TypeError: If tags is not a dictionary.
        """

        if not isinstance(tags, dict):
            raise TypeError("tags must be a dictionary.")

        normalized_tags = {
            str(key): str(value)
            for key, value in tags.items()
        }

        mlflow.set_tags(normalized_tags)

        self.logger.info(
            "Logged %d tags to MLflow.",
            len(normalized_tags),
        )

    def log_metrics(
        self,
        metrics: dict[str, float],
    ) -> None:
        """
        Log evaluation metrics to the active MLflow run.

        Args:
            metrics: Mapping of metric names to numeric values.

        Raises:
            TypeError: If metrics is not a dictionary.
            ValueError: If a metric value is not numeric.
        """

        if not isinstance(metrics, dict):
            raise TypeError(
                "metrics must be a dictionary."
            )

        normalized_metrics: dict[str, float] = {}

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Metric '{key}' must have a numeric value."
                )

            normalized_metrics[str(key)] = float(value)

        mlflow.log_metrics(normalized_metrics)

        self.logger.info(
            "Logged %d metrics to MLflow.",
            len(normalized_metrics),
        )

    def log_artifact(
        self,
        artifact_path: str | Path,
    ) -> None:
        """
        Log a persisted file as an MLflow artifact.

        Args:
            artifact_path: Path to the artifact file.

        Raises:
            FileNotFoundError: If the artifact does not exist.
            ValueError: If the artifact path is not a file.
        """

        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Artifact does not exist: {artifact_path}"
            )

        if not artifact_path.is_file():
            raise ValueError(
                f"Artifact path must be a file: {artifact_path}"
            )

        mlflow.log_artifact(
            str(artifact_path)
        )

        self.logger.info(
            "Logged MLflow artifact: %s",
            artifact_path,
        )

    def log_artifacts(
        self,
        artifact_directory: str | Path,
    ) -> None:
        """
        Log all files from a directory as MLflow artifacts.

        Args:
            artifact_directory: Directory containing artifacts.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If the path is not a directory.
        """

        artifact_directory = Path(artifact_directory)

        if not artifact_directory.exists():
            raise FileNotFoundError(
                f"Artifact directory does not exist: "
                f"{artifact_directory}"
            )

        if not artifact_directory.is_dir():
            raise ValueError(
                f"Artifact path must be a directory: "
                f"{artifact_directory}"
            )

        mlflow.log_artifacts(
            str(artifact_directory)
        )

        self.logger.info(
            "Logged MLflow artifact directory: %s",
            artifact_directory,
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the active MLflow run.

        Args:
            status: Final MLflow run status.
                    Typically 'FINISHED' or 'FAILED'.
        """

        active_run = mlflow.active_run()

        if active_run is None:
            self.logger.warning(
                "No active MLflow run exists to end."
            )
            return

        run_id = active_run.info.run_id

        mlflow.end_run(status=status)

        self.logger.info(
            "MLflow run ended | run_id=%s | status=%s",
            run_id,
            status,
        )
        
    @staticmethod
    def _normalize_parameter_value(
        value: Any,
    ) -> str:
        """
        Convert a parameter value into an MLflow-compatible string.

        Args:
            value: Parameter value.

        Returns:
            String representation of the parameter.
        """

        if isinstance(value, (dict, list, tuple)):
            return str(value)

        return str(value)