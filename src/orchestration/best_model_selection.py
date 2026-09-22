# src/orchestration/best_model_selection.py

"""Best-model selection based on completed model comparison results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class BestModelSelection:
    """
    Select the best model from completed comparison results.

    The selection decision is controlled entirely by the model-selection
    policy supplied by the application configuration.

    This component does not:
        - train models
        - compare models
        - run inference
        - modify candidate model artifacts

    It only:
        - reads completed model comparison results
        - applies the configured selection policy
        - validates the selected model artifact
        - returns the selected model information
    """

    SUPPORTED_METRICS = (
        "accuracy",
        "precision",
        "recall",
        "f1",
    )

    SUPPORTED_DIRECTIONS = (
        "maximize",
        "minimize",
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
    )

    def __init__(
        self,
        selection_config: dict[str, Any],
        comparison_results_path: str | Path,
    ) -> None:
        """
        Initialize the best-model selection component.

        Args:
            selection_config: Model-selection policy loaded from YAML.
            comparison_results_path: Path to the completed model comparison
                CSV file.

        Raises:
            TypeError: If selection_config is not a dictionary.
            ValueError: If the selection policy or comparison results path
                is invalid.
        """

        if not isinstance(selection_config, dict):
            raise TypeError(
                "selection_config must be a dictionary."
            )

        self.selection_config = selection_config
        self.comparison_results_path = Path(
            comparison_results_path
        )

        self.enabled = self._validate_enabled()
        self.metric = self._validate_metric()
        self.direction = self._validate_direction()

        logger.info(
            "Best-model selection initialized: "
            "enabled=%s, metric=%s, direction=%s, "
            "comparison_results_path=%s",
            self.enabled,
            self.metric,
            self.direction,
            self.comparison_results_path,
        )

    def select(self) -> dict[str, Any]:
        """
        Select the best model from persisted comparison results.

        Returns:
            Metadata for the selected model.

        Raises:
            ValueError: If selection is disabled, comparison results are
                invalid, required metrics are missing, or an artifact is
                invalid.
            FileNotFoundError: If the comparison results file or selected
                model artifact does not exist.
            RuntimeError: If the comparison CSV cannot be read.
        """

        if not self.enabled:
            raise ValueError(
                "Best-model selection is disabled in configuration."
            )

        comparison_results = self._load_comparison_results()

        self._validate_comparison_results(
            comparison_results
        )

        selected_model = self._select_model(
            comparison_results
        )

        selected_result = comparison_results[selected_model]

        model_artifact = self._resolve_model_artifact(
            selected_model=selected_model,
            selected_result=selected_result,
        )

        selected_score = selected_result[self.metric]

        selection_result = {
            "model": selected_model,
            "metric": self.metric,
            "direction": self.direction,
            "score": selected_score,
            "model_artifact": str(model_artifact),
        }

        logger.info(
            "Best model selected: model=%s, metric=%s, "
            "score=%.6f, artifact=%s",
            selected_model,
            self.metric,
            selected_score,
            model_artifact,
        )

        return selection_result

    def _load_comparison_results(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Load completed model comparison results from CSV.

        Returns:
            Dictionary mapping each model name to its comparison metadata.

        Raises:
            FileNotFoundError: If the comparison CSV does not exist.
            RuntimeError: If the CSV cannot be read or contains invalid
                metric values.
        """

        if not self.comparison_results_path.exists():
            raise FileNotFoundError(
                "Model comparison results file does not exist: "
                f"{self.comparison_results_path}"
            )

        if not self.comparison_results_path.is_file():
            raise ValueError(
                "Model comparison results path must be a file: "
                f"{self.comparison_results_path}"
            )

        comparison_results: dict[str, dict[str, Any]] = {}

        logger.info(
            "Loading model comparison results from: %s",
            self.comparison_results_path,
        )

        try:
            with self.comparison_results_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                reader = csv.DictReader(csv_file)

                if reader.fieldnames is None:
                    raise ValueError(
                        "Model comparison CSV does not contain a header."
                    )

                missing_columns = set(
                    self.COMPARISON_COLUMNS
                ) - set(reader.fieldnames)

                if missing_columns:
                    raise ValueError(
                        "Model comparison CSV is missing required "
                        f"columns: {sorted(missing_columns)}"
                    )

                for row_number, row in enumerate(
                    reader,
                    start=2,
                ):
                    model_name = row.get("model")

                    if not model_name:
                        raise ValueError(
                            "Model name is missing in comparison CSV "
                            f"row {row_number}."
                        )

                    try:
                        comparison_results[model_name] = {
                            "accuracy": float(
                                row["accuracy"]
                            ),
                            "precision": float(
                                row["precision"]
                            ),
                            "recall": float(
                                row["recall"]
                            ),
                            "f1": float(
                                row["f1"]
                            ),
                            "training_rows": int(
                                row["training_rows"]
                            ),
                            "validation_rows": int(
                                row["validation_rows"]
                            ),
                            "feature_count": int(
                                row["feature_count"]
                            ),
                            "model_artifact": row[
                                "model_artifact"
                            ],
                        }

                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Invalid comparison metric value in "
                            f"row {row_number} for model "
                            f"'{model_name}'."
                        ) from exc

        except OSError as exc:
            logger.exception(
                "Failed to read model comparison results."
            )
            raise RuntimeError(
                "Failed to read model comparison results."
            ) from exc

        logger.info(
            "Loaded comparison results for %d models.",
            len(comparison_results),
        )

        return comparison_results

    def _select_model(
        self,
        comparison_results: dict[str, dict[str, Any]],
    ) -> str:
        """
        Select a model according to the configured metric and direction.

        Args:
            comparison_results: Completed model comparison results.

        Returns:
            Name of the selected model.
        """

        if self.direction == "maximize":
            return max(
                comparison_results,
                key=lambda model: comparison_results[model][
                    self.metric
                ],
            )

        return min(
            comparison_results,
            key=lambda model: comparison_results[model][
                self.metric
            ],
        )

    def _resolve_model_artifact(
        self,
        selected_model: str,
        selected_result: dict[str, Any],
    ) -> Path:
        """
        Resolve and validate the selected model artifact.

        Args:
            selected_model: Name of the selected model.
            selected_result: Comparison metadata for the selected model.

        Returns:
            Path to the trained model artifact.

        Raises:
            ValueError: If the artifact path is missing or invalid.
            FileNotFoundError: If the artifact does not exist.
        """

        artifact_value = selected_result.get(
            "model_artifact"
        )

        if not artifact_value:
            raise ValueError(
                "Model artifact path is missing for selected model "
                f"'{selected_model}'."
            )

        artifact_path = Path(artifact_value)

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact for '{selected_model}' does not exist: "
                f"{artifact_path}"
            )

        if not artifact_path.is_file():
            raise ValueError(
                f"Model artifact is not a file: {artifact_path}"
            )

        return artifact_path

    def _validate_enabled(self) -> bool:
        """Validate and return the selection-enabled flag."""

        enabled = self.selection_config.get(
            "enabled",
            True,
        )

        if not isinstance(enabled, bool):
            raise ValueError(
                "'model_selection.enabled' must be a boolean."
            )

        return enabled

    def _validate_metric(self) -> str:
        """Validate and return the configured selection metric."""

        metric = self.selection_config.get("metric")

        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported model-selection metric: {metric}. "
                f"Supported metrics: {self.SUPPORTED_METRICS}"
            )

        return metric

    def _validate_direction(self) -> str:
        """Validate and return the configured selection direction."""

        direction = self.selection_config.get("direction")

        if direction not in self.SUPPORTED_DIRECTIONS:
            raise ValueError(
                f"Unsupported model-selection direction: {direction}. "
                f"Supported directions: {self.SUPPORTED_DIRECTIONS}"
            )

        return direction

    @staticmethod
    def _validate_comparison_results(
        comparison_results: dict[str, dict[str, Any]],
    ) -> None:
        """
        Validate the structure of comparison results.

        Args:
            comparison_results: Results loaded from the comparison CSV.

        Raises:
            TypeError: If the results have an invalid structure.
            ValueError: If the results are empty or required metrics are
                missing.
        """

        if not isinstance(comparison_results, dict):
            raise TypeError(
                "comparison_results must be a dictionary."
            )

        if not comparison_results:
            raise ValueError(
                "comparison_results cannot be empty."
            )

        for model_name, result in comparison_results.items():
            if not isinstance(model_name, str):
                raise TypeError(
                    "Each model name in comparison_results "
                    "must be a string."
                )

            if not isinstance(result, dict):
                raise TypeError(
                    f"Comparison result for '{model_name}' "
                    "must be a dictionary."
                )

            for metric in BestModelSelection.SUPPORTED_METRICS:
                if metric not in result:
                    raise ValueError(
                        f"Metric '{metric}' is missing for model "
                        f"'{model_name}'."
                    )

                if not isinstance(
                    result[metric],
                    (int, float),
                ):
                    raise TypeError(
                        f"Metric '{metric}' for model "
                        f"'{model_name}' must be numeric."
                    )

            if not result.get("model_artifact"):
                raise ValueError(
                    f"Model artifact is missing for model "
                    f"'{model_name}'."
                )