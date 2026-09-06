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
"""

import logging
from pathlib import Path

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
    ) -> None:
        """
        Initialize the feature engineering pipeline.

        Args:
            input_path: Path to the processed CSV dataset.
            output_dir: Directory where feature matrices and labels are saved.
            vectorizer_path: Path where the fitted text vectorizer is saved.
            random_state: Seed used for reproducible dataset splitting.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If paths are empty or random_state is invalid.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if not isinstance(input_path, (str, Path)):
            raise TypeError("input_path must be a string or Path.")

        if not isinstance(output_dir, (str, Path)):
            raise TypeError("output_dir must be a string or Path.")

        if not isinstance(vectorizer_path, (str, Path)):
            raise TypeError("vectorizer_path must be a string or Path.")

        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")

        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.vectorizer_path = Path(vectorizer_path)
        self.random_state = random_state

    @property
    def name(self) -> str:
        """
        Return the registered pipeline name.

        Returns:
            str: Pipeline identifier.
        """

        return self.PIPELINE_NAME

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
            raise TypeError("dataframe must be a pandas DataFrame.")

        required_columns = {"Email Text", "label"}
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                "Processed dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if dataframe.empty:
            raise ValueError("Processed dataset is empty.")

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

        Split ratio:
            Training   = 70%
            Validation = 15%
            Test       = 15%

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

        # First split:
        # 70% training
        # 30% temporary set
        (
            training_texts,
            temporary_texts,
            training_labels,
            temporary_labels,
        ) = train_test_split(
            texts,
            labels,
            test_size=0.30,
            random_state=self.random_state,
            stratify=labels,
        )

        # Second split:
        # Split the 30% temporary set equally:
        # 15% validation
        # 15% test
        (
            validation_texts,
            test_texts,
            validation_labels,
            test_labels,
        ) = train_test_split(
            temporary_texts,
            temporary_labels,
            test_size=0.50,
            random_state=self.random_state,
            stratify=temporary_labels,
        )

        self.logger.info(
            "Dataset split completed: train=%d, validation=%d, test=%d.",
            len(training_texts),
            len(validation_texts),
            len(test_texts),
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

        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        self.logger.info("Starting feature engineering pipeline.")

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

        extractor = TextFeatureExtractor()

        # IMPORTANT:
        # TF-IDF is fitted ONLY on training text.
        training_features = extractor.fit_transform(training_texts)

        # Validation and test data are transformed using the already-fitted
        # training vectorizer. No fitting occurs on either dataset.
        validation_features = extractor.transform(validation_texts)
        test_features = extractor.transform(test_texts)

        extractor.save(self.vectorizer_path)

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
            },
        )