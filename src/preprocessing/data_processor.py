#src/preprocessing/data_processor.py

"""
Data preprocessing module for the phishing email classifier.

This module is responsible for transforming the raw phishing email
dataset into a clean, validated dataset suitable for downstream
feature engineering and model training.

The raw dataset is never modified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


class DataProcessor:
    """
    Handles preprocessing and validation of the phishing email dataset.

    The processor:
        1. Loads the raw CSV file.
        2. Validates the expected schema.
        3. Removes the unnecessary index column.
        4. Removes emails with missing/empty text.
        5. Normalizes target labels.
        6. Removes duplicate email texts.
        7. Validates the resulting dataset.
        8. Saves the processed dataset.

    Attributes:
        input_path: Path to the raw CSV dataset.
        output_path: Path where the processed dataset will be saved.
    """

    REQUIRED_COLUMNS = {"Email Text", "Email Type"}

    LABEL_MAPPING = {
        "Safe Email": 0,
        "Phishing Email": 1,
    }

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """
        Initialize the DataProcessor.

        Args:
            input_path: Path to the raw CSV file.
            output_path: Path for the processed CSV file.

        Raises:
            ValueError: If input_path and output_path refer to the same file.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError(
                "Input and output paths must be different. "
                "The raw dataset must never be overwritten."
            )

        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset from CSV.

        Returns:
            DataFrame containing the raw dataset.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the dataset is empty.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {self.input_path}"
            )

        self.logger.info("Loading dataset: %s", self.input_path)

        dataframe = pd.read_csv(self.input_path)

        if dataframe.empty:
            raise ValueError("Input dataset is empty.")

        self.logger.info(
            "Dataset loaded successfully: %d rows, %d columns",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    def validate_schema(self, dataframe: pd.DataFrame) -> None:
        """
        Validate that the dataset contains the required columns.

        Args:
            dataframe: Dataset to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        missing_columns = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        self.logger.info("Schema validation passed.")

    def remove_index_column(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the unnecessary CSV index column if present.

        Args:
            dataframe: Dataset to process.

        Returns:
            DataFrame without the index column.
        """
        dataframe = dataframe.copy()

        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(columns=["Unnamed: 0"])
            self.logger.info("Removed 'Unnamed: 0' index column.")

        return dataframe

    def clean_email_text(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows containing missing or empty email text.

        Whitespace-only emails are considered empty.

        Args:
            dataframe: Dataset to process.

        Returns:
            DataFrame containing only rows with usable email text.
        """
        dataframe = dataframe.copy()

        dataframe["Email Text"] = dataframe["Email Text"].fillna("").astype(str)

        dataframe["Email Text"] = dataframe["Email Text"].str.strip()

        before_count = len(dataframe)

        dataframe = dataframe[dataframe["Email Text"] != ""].copy()

        removed_count = before_count - len(dataframe)

        self.logger.info(
            "Removed %d rows with missing/empty email text.",
            removed_count,
        )

        return dataframe

    def encode_labels(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Convert textual labels into binary numerical labels.

        Mapping:
            Safe Email -> 0
            Phishing Email -> 1

        Args:
            dataframe: Dataset to process.

        Returns:
            DataFrame with numerical target labels.

        Raises:
            ValueError: If an unknown label is encountered.
        """
        dataframe = dataframe.copy()

        unknown_labels = set(dataframe["Email Type"].unique()) - set(
            self.LABEL_MAPPING
        )

        if unknown_labels:
            raise ValueError(
                f"Unknown email labels detected: {sorted(unknown_labels)}"
            )

        dataframe["Email Type"] = dataframe["Email Type"].map(
            self.LABEL_MAPPING
        )

        dataframe = dataframe.rename(columns={"Email Type": "label"})

        self.logger.info("Labels encoded successfully.")

        return dataframe

    def remove_duplicates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate email texts.

        Only the first occurrence of each email is retained.

        Args:
            dataframe: Dataset to process.

        Returns:
            DataFrame with unique email texts.
        """
        dataframe = dataframe.copy()

        before_count = len(dataframe)

        dataframe = dataframe.drop_duplicates(
            subset=["Email Text"],
            keep="first",
        ).reset_index(drop=True)

        removed_count = before_count - len(dataframe)

        self.logger.info(
            "Removed %d duplicate email texts.",
            removed_count,
        )

        return dataframe

    def validate_processed_data(self, dataframe: pd.DataFrame) -> None:
        """
        Validate the final processed dataset.

        Args:
            dataframe: Processed dataset.

        Raises:
            ValueError: If validation fails.
        """
        if dataframe.empty:
            raise ValueError("Processed dataset is empty.")

        if dataframe["Email Text"].isna().any():
            raise ValueError("Processed dataset contains missing emails.")

        if dataframe["Email Text"].str.strip().eq("").any():
            raise ValueError("Processed dataset contains empty emails.")

        if dataframe["Email Text"].duplicated().any():
            raise ValueError("Processed dataset contains duplicate emails.")

        if not dataframe["label"].isin(self.LABEL_MAPPING.values()).all():
            raise ValueError("Processed dataset contains invalid labels.")

        self.logger.info("Processed dataset validation passed.")

    def save_data(self, dataframe: pd.DataFrame) -> None:
        """
        Save the processed dataset to disk.

        Args:
            dataframe: Processed dataset.
        """
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            self.output_path,
            index=False,
        )

        self.logger.info(
            "Processed dataset saved to: %s",
            self.output_path,
        )

    def process(self) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.

        Returns:
            Cleaned and validated DataFrame.
        """
        self.logger.info("Starting data preprocessing pipeline.")

        dataframe = self.load_data()

        self.validate_schema(dataframe)

        dataframe = self.remove_index_column(dataframe)

        dataframe = self.clean_email_text(dataframe)

        dataframe = self.encode_labels(dataframe)

        dataframe = self.remove_duplicates(dataframe)

        self.validate_processed_data(dataframe)

        self.save_data(dataframe)

        self.logger.info(
            "Preprocessing pipeline completed successfully. "
            "Final dataset size: %d rows.",
            len(dataframe),
        )

        return dataframe





