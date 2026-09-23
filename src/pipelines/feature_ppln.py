# src/pipelines/feature_ppln.py

"""
Feature engineering pipeline for the phishing email classifier.

This module is responsible for:
    1. Loading the processed dataset.
    2. Validating the required schema.
    3. Splitting the dataset into training, validation, and test sets.
    4. Fitting the text feature extractor only on training data.
    5. Transforming validation and test data using the fitted extractor.
    6. Persisting feature matrices, labels, and the fitted vectorizer.

The pipeline does not train a machine learning model.
Model training is handled separately by TrainingPipeline.

Feature extraction and dataset split parameters are supplied by the
application configuration layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.model_selection import train_test_split

from src.features.text_feature_extractor import TextFeatureExtractor
from src.orchestration.base_ppln import BasePipeline, PipelineResult


class FeaturePipeline(BasePipeline):
    """
    Pipeline responsible for dataset splitting and text feature extraction.

    The TF-IDF extractor is fitted exclusively on the training dataset to
    prevent validation/test data leakage.
    """

    PIPELINE_NAME = "features"

    def __init__(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        vectorizer_path: str | Path,
        random_state: int = 42,
        tfidf_config: dict[str, Any] | None = None,
        split_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the feature engineering pipeline.

        Args:
            input_path: Path to the processed CSV dataset.
            output_dir: Directory where feature matrices and labels are saved.
            vectorizer_path: Path where the fitted text vectorizer is saved.
            random_state: Seed used for reproducible dataset splitting.
            tfidf_config: Configuration for TF-IDF feature extraction.
            split_config: Configuration for train/validation/test splitting.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If paths, random_state, or configuration values
                are invalid.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if not isinstance(input_path, (str, Path)):
            raise TypeError("input_path must be a string or Path.")

        if not isinstance(output_dir, (str, Path)):
            raise TypeError("output_dir must be a string or Path.")

        if not isinstance(vectorizer_path, (str, Path)):
            raise TypeError(
                "vectorizer_path must be a string or Path."
            )

        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")

        if tfidf_config is None:
            raise ValueError(
                "tfidf_config must be provided."
            )

        if not isinstance(tfidf_config, dict):
            raise TypeError(
                "tfidf_config must be a dictionary."
            )

        if split_config is None:
            raise ValueError(
                "split_config must be provided."
            )

        if not isinstance(split_config, dict):
            raise TypeError(
                "split_config must be a dictionary."
            )

        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.vectorizer_path = Path(vectorizer_path)
        self.random_state = random_state

        self.tfidf_config = self._validate_tfidf_config(
            tfidf_config
        )

        self.split_config = self._validate_split_config(
            split_config
        )

    @property
    def name(self) -> str:
        """
        Return the registered pipeline name.

        Returns:
            str: Pipeline identifier.
        """

        return self.PIPELINE_NAME

    @staticmethod
    def _validate_tfidf_config(
        tfidf_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate the TF-IDF configuration required by TextFeatureExtractor.

        Args:
            tfidf_config: TF-IDF configuration mapping.

        Returns:
            dict[str, Any]: Validated TF-IDF configuration.

        Raises:
            ValueError: If required configuration values are missing.
        """

        required_parameters = (
            "max_features",
            "ngram_range",
            "min_df",
            "max_df",
        )

        missing_parameters = [
            parameter
            for parameter in required_parameters
            if parameter not in tfidf_config
        ]

        if missing_parameters:
            raise ValueError(
                "Missing required TF-IDF configuration parameter(s): "
                f"{missing_parameters}"
            )

        return dict(tfidf_config)

    @staticmethod
    def _validate_split_config(
        split_config: dict[str, Any],
    ) -> dict[str, float]:
        """
        Validate the final train/validation/test split proportions.

        Args:
            split_config: Dataset split configuration.

        Returns:
            dict[str, float]: Validated split proportions.

        Raises:
            ValueError: If required values are missing, invalid, or do not
                sum to one.
            TypeError: If split values are not numeric.
        """

        required_sizes = (
            "train_size",
            "validation_size",
            "test_size",
        )

        missing_sizes = [
            size
            for size in required_sizes
            if size not in split_config
        ]

        if missing_sizes:
            raise ValueError(
                "Missing required split configuration value(s): "
                f"{missing_sizes}"
            )

        sizes: dict[str, float] = {}

        for size_name in required_sizes:
            value = split_config[size_name]

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{size_name} must be a numeric value."
                )

            value = float(value)

            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"{size_name} must be greater than 0 and less than 1."
                )

            sizes[size_name] = value

        total = sum(sizes.values())

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "train_size, validation_size, and test_size "
                f"must sum to 1.0. Received {total:.6f}."
            )

        return sizes

    def _load_data(self) -> pd.DataFrame:
        """
        Load the processed dataset from disk.

        Returns:
            pd.DataFrame: Processed phishing email dataset.

        Raises:
            FileNotFoundError: If the input dataset does not exist.
            ValueError: If the input path is not a file.
        """

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.input_path}"
            )

        if not self.input_path.is_file():
            raise ValueError(
                f"Input path is not a file: {self.input_path}"
            )

        self.logger.info(
            "Loading processed dataset from: %s",
            self.input_path,
        )

        dataframe = pd.read_csv(self.input_path)

        self.logger.info(
            "Loaded processed dataset with %d rows.",
            len(dataframe),
        )

        return dataframe

    @staticmethod
    def _validate_schema(dataframe: pd.DataFrame) -> None:
        """
        Validate that the processed dataset contains the required columns.

        Args:
            dataframe: Dataset to validate.

        Raises:
            TypeError: If dataframe is not a pandas DataFrame.
            ValueError: If required columns are missing.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        required_columns = {"Email Text", "label"}
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                "Processed dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if dataframe.empty:
            raise ValueError(
                "Processed dataset is empty."
            )

    def _split_data(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """
        Split the dataset into training, validation, and test sets.

        The split is stratified using the target label so that class
        proportions remain approximately consistent across all datasets.

        The final proportions are controlled by split configuration.

        Args:
            dataframe: Validated processed dataset.

        Returns:
            Tuple containing:
                training_texts,
                validation_texts,
                test_texts,
                training_labels,
                validation_labels,
                test_labels.
        """

        texts = dataframe["Email Text"]
        labels = dataframe["label"]

        train_size = self.split_config["train_size"]
        validation_size = self.split_config["validation_size"]
        test_size = self.split_config["test_size"]

        # The first split separates the configured training proportion
        # from the combined validation + test proportion.
        temporary_size = validation_size + test_size

        (
            training_texts,
            temporary_texts,
            training_labels,
            temporary_labels,
        ) = train_test_split(
            texts,
            labels,
            test_size=temporary_size,
            random_state=self.random_state,
            stratify=labels,
        )

        # The second split divides the temporary dataset according to
        # the relative proportions of validation and test data.
        test_relative_size = test_size / temporary_size

        (
            validation_texts,
            test_texts,
            validation_labels,
            test_labels,
        ) = train_test_split(
            temporary_texts,
            temporary_labels,
            test_size=test_relative_size,
            random_state=self.random_state,
            stratify=temporary_labels,
        )

        self.logger.info(
            "Dataset split completed: train=%d (%.2f%%), "
            "validation=%d (%.2f%%), test=%d (%.2f%%).",
            len(training_texts),
            train_size * 100,
            len(validation_texts),
            validation_size * 100,
            len(test_texts),
            test_size * 100,
        )

        return (
            training_texts,
            validation_texts,
            test_texts,
            training_labels,
            validation_labels,
            test_labels,
        )

    def _save_features(
        self,
        training_features,
        validation_features,
        test_features,
        training_labels: pd.Series,
        validation_labels: pd.Series,
        test_labels: pd.Series,
    ) -> None:
        """
        Persist feature matrices and labels to disk.

        Sparse matrices are saved using scipy's NPZ format because TF-IDF
        matrices can contain a large number of zero values.

        Args:
            training_features: Sparse training feature matrix.
            validation_features: Sparse validation feature matrix.
            test_features: Sparse test feature matrix.
            training_labels: Training target labels.
            validation_labels: Validation target labels.
            test_labels: Test target labels.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_npz(
            self.output_dir / "X_train.npz",
            training_features,
        )

        save_npz(
            self.output_dir / "X_validation.npz",
            validation_features,
        )

        save_npz(
            self.output_dir / "X_test.npz",
            test_features,
        )

        joblib.dump(
            training_labels.to_numpy(),
            self.output_dir / "y_train.joblib",
        )

        joblib.dump(
            validation_labels.to_numpy(),
            self.output_dir / "y_validation.joblib",
        )

        joblib.dump(
            test_labels.to_numpy(),
            self.output_dir / "y_test.joblib",
        )

        self.logger.info(
            "Feature matrices and labels saved to: %s",
            self.output_dir,
        )

    def run(self) -> PipelineResult:
        """
        Execute the complete feature engineering workflow.

        Returns:
            PipelineResult: Standardized result describing pipeline execution.

        Raises:
            RuntimeError: If feature engineering fails.
        """

        self.logger.info(
            "Starting feature engineering pipeline."
        )

        dataframe = self._load_data()
        self._validate_schema(dataframe)

        (
            training_texts,
            validation_texts,
            test_texts,
            training_labels,
            validation_labels,
            test_labels,
        ) = self._split_data(dataframe)

        extractor = TextFeatureExtractor(
            max_features=self.tfidf_config["max_features"],
            ngram_range=self.tfidf_config["ngram_range"],
            min_df=self.tfidf_config["min_df"],
            max_df=self.tfidf_config["max_df"],
        )

        # IMPORTANT:
        # TF-IDF is fitted ONLY on training text.
        training_features = extractor.fit_transform(
            training_texts
        )

        # Validation and test data are transformed using the already-fitted
        # training vectorizer. No fitting occurs on either dataset.
        validation_features = extractor.transform(
            validation_texts
        )

        test_features = extractor.transform(
            test_texts
        )

        extractor.save(
            self.vectorizer_path
        )

        self._save_features(
            training_features=training_features,
            validation_features=validation_features,
            test_features=test_features,
            training_labels=training_labels,
            validation_labels=validation_labels,
            test_labels=test_labels,
        )

        self.logger.info(
            "Feature engineering pipeline completed successfully."
        )

        return PipelineResult(
            pipeline_name=self.name,
            success=True,
            message="Feature engineering completed successfully.",
            metadata={
                "input_path": str(self.input_path),
                "output_dir": str(self.output_dir),
                "vectorizer_path": str(self.vectorizer_path),
                "train_rows": len(training_texts),
                "validation_rows": len(validation_texts),
                "test_rows": len(test_texts),
                "feature_count": training_features.shape[1],
                "split_config": self.split_config,
                "tfidf_config": self.tfidf_config,
            },
        )