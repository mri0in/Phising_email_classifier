# src/pipelines/evaluation_ppln.py

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import load_npz

from src.evaluation.evaluator import Evaluator
from src.orchestration.base_ppln import BasePipeline, PipelineResult


class EvaluationPipeline(BasePipeline):
    """
    Evaluate the trained phishing email classifier on the untouched
    test dataset.

    The pipeline:
        1. Loads X_test and y_test.
        2. Loads the trained model.
        3. Generates predictions on X_test.
        4. Calculates final evaluation metrics.
        5. Saves the metrics to disk.

    The test dataset is never used for training or model fitting.
    """

    PIPELINE_NAME = "evaluation"

    def __init__(
        self,
        features_dir: str | Path,
        model_path: str | Path,
        metrics_output_path: str | Path,
    ) -> None:
        """
        Initialize the evaluation pipeline.

        Args:
            features_dir: Directory containing test feature artifacts.
            model_path: Path to the trained classifier.
            metrics_output_path: Path where evaluation metrics are saved.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        self.features_dir = Path(features_dir)
        self.model_path = Path(model_path)
        self.metrics_output_path = Path(metrics_output_path)

        self._validate_paths()

    @property
    def name(self) -> str:
        """Return the registered pipeline name."""

        return self.PIPELINE_NAME

    def _validate_paths(self) -> None:
        """Validate required paths and create output directories."""

        if not self.features_dir.exists():
            raise FileNotFoundError(
                f"Features directory does not exist: {self.features_dir}"
            )

        if not self.features_dir.is_dir():
            raise NotADirectoryError(
                f"Features path is not a directory: {self.features_dir}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file does not exist: {self.model_path}"
            )

        if not self.model_path.is_file():
            raise ValueError(
                f"Model path is not a file: {self.model_path}"
            )

        self.metrics_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _load_test_data(self) -> tuple[Any, np.ndarray]:
        """
        Load the untouched test feature matrix and labels.

        Returns:
            Tuple containing X_test and y_test.
        """

        x_test_path = self.features_dir / "X_test.npz"
        y_test_path = self.features_dir / "y_test.joblib"

        if not x_test_path.exists():
            raise FileNotFoundError(
                f"Test feature file does not exist: {x_test_path}"
            )

        if not y_test_path.exists():
            raise FileNotFoundError(
                f"Test label file does not exist: {y_test_path}"
            )

        self.logger.info("Loading untouched test dataset.")

        X_test = load_npz(x_test_path)
        y_test = joblib.load(y_test_path)

        if not isinstance(y_test, np.ndarray):
            y_test = np.asarray(y_test)

        self.logger.info(
            "Test dataset loaded: %d samples, %d features.",
            X_test.shape[0],
            X_test.shape[1],
        )

        return X_test, y_test

    def _load_model(self) -> Any:
        """
        Load the trained classifier from disk.

        Returns:
            Trained model object.
        """

        self.logger.info(
            "Loading trained model from: %s",
            self.model_path,
        )

        model = joblib.load(self.model_path)

        if not hasattr(model, "predict"):
            raise TypeError(
                "Loaded model does not implement predict()."
            )

        return model

    @staticmethod
    def _validate_test_data(
        X_test: Any,
        y_test: np.ndarray,
    ) -> None:
        """
        Validate the test feature matrix and labels.

        Args:
            X_test: Test feature matrix.
            y_test: Test labels.

        Raises:
            ValueError: If dimensions or sample counts are invalid.
        """

        if X_test.shape[0] == 0:
            raise ValueError("X_test cannot be empty.")

        if y_test.size == 0:
            raise ValueError("y_test cannot be empty.")

        if X_test.shape[0] != len(y_test):
            raise ValueError(
                "X_test and y_test must contain the same number of samples."
            )

        if len(np.unique(y_test)) < 2:
            raise ValueError(
                "y_test must contain at least two classes."
            )

    def _save_metrics(
        self,
        metrics: dict[str, Any],
    ) -> None:
        """
        Save evaluation metrics to disk.

        Args:
            metrics: Evaluation metrics dictionary.
        """

        self.logger.info(
            "Saving evaluation metrics to: %s",
            self.metrics_output_path,
        )

        joblib.dump(
            metrics,
            self.metrics_output_path,
        )

    def run(self) -> PipelineResult:
        """
        Execute the evaluation pipeline.

        Returns:
            PipelineResult containing final test-set metrics.
        """

        self.logger.info(
            "Starting evaluation pipeline."
        )

        try:
            X_test, y_test = self._load_test_data()

            self._validate_test_data(
                X_test,
                y_test,
            )

            model = self._load_model()

            self.logger.info(
                "Generating predictions on untouched test data."
            )

            y_pred = model.predict(X_test)

            evaluator = Evaluator()

            metrics = evaluator.evaluate(
                y_true=y_test,
                y_pred=y_pred,
            )

            self._save_metrics(metrics)

            self.logger.info(
                "Evaluation pipeline completed successfully."
            )

            return PipelineResult(
                pipeline_name=self.name,
                success=True,
                message="Model evaluation completed successfully.",
                metadata={
                    "test_rows": len(y_test),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "confusion_matrix": metrics[
                        "confusion_matrix"
                    ],
                    "metrics_path": str(
                        self.metrics_output_path
                    ),
                },
            )

        except Exception as exc:
            self.logger.exception(
                "Evaluation pipeline failed."
            )

            raise RuntimeError(
                "Evaluation pipeline failed."
            ) from exc