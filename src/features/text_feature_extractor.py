# src/features/text_feature_extractor.py

"""
Text feature extraction using TF-IDF.

This module provides a reusable component for converting email text
into numerical TF-IDF feature representations.

Responsibilities:
    - Validate text input.
    - Fit a TF-IDF vectorizer on training text.
    - Transform text using the fitted vectorizer.
    - Fit and transform training text when required.
    - Persist and load the fitted vectorizer.

The component does not decide how data is split or when the model
should be trained. Those responsibilities belong to the pipeline layer.
"""

import logging
from pathlib import Path
from typing import Iterable

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    """
    Extract numerical TF-IDF features from email text.

    The extractor maintains a fitted TfidfVectorizer so that the exact
    same vocabulary and weighting scheme can be reused during
    validation, testing, and inference.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> None:
        """
        Initialize the text feature extractor.

        Args:
            max_features: Maximum number of TF-IDF features to retain.
            ngram_range: Lower and upper boundaries for n-grams.
            min_df: Minimum document frequency for a term.
            max_df: Maximum document frequency for a term.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If an argument has an invalid value.
        """

        if not isinstance(max_features, int):
            raise TypeError("max_features must be an integer.")

        if max_features <= 0:
            raise ValueError("max_features must be greater than zero.")

        if (
            not isinstance(ngram_range, tuple)
            or len(ngram_range) != 2
            or not all(isinstance(value, int) for value in ngram_range)
        ):
            raise TypeError(
                "ngram_range must be a tuple containing two integers."
            )

        if ngram_range[0] <= 0 or ngram_range[1] < ngram_range[0]:
            raise ValueError(
                "ngram_range must contain valid positive boundaries."
            )

        if not isinstance(min_df, int):
            raise TypeError("min_df must be an integer.")

        if min_df <= 0:
            raise ValueError("min_df must be greater than zero.")

        if not isinstance(max_df, float):
            raise TypeError("max_df must be a float.")

        if not 0.0 < max_df <= 1.0:
            raise ValueError("max_df must be between 0 and 1.")

        self.logger = logging.getLogger(self.__class__.__name__)

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

        self._is_fitted = False

    @staticmethod
    def _validate_texts(texts: Iterable[str]) -> list[str]:
        """
        Validate and normalize a collection of text documents.

        Args:
            texts: Collection of email text documents.

        Returns:
            list[str]: Validated text documents.

        Raises:
            TypeError: If texts or an individual document is invalid.
            ValueError: If no valid documents are supplied.
        """

        if isinstance(texts, (str, bytes)):
            raise TypeError(
                "texts must be an iterable of documents, not a single string."
            )

        try:
            documents = list(texts)
        except TypeError as exc:
            raise TypeError(
                "texts must be an iterable of strings."
            ) from exc

        if not documents:
            raise ValueError("At least one text document is required.")

        validated_documents: list[str] = []

        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(
                    f"Document at index {index} must be a string."
                )

            cleaned_document = document.strip()

            if not cleaned_document:
                raise ValueError(
                    f"Document at index {index} cannot be empty."
                )

            validated_documents.append(cleaned_document)

        return validated_documents

    def fit(self, training_texts: Iterable[str]) -> None:
        """
        Fit the TF-IDF vectorizer using training text only.

        Args:
            training_texts: Email text documents from the training set.

        Raises:
            TypeError: If training_texts is invalid.
            ValueError: If training_texts contains invalid documents.
        """

        documents = self._validate_texts(training_texts)

        self.logger.info(
            "Fitting TF-IDF vectorizer on %d training documents.",
            len(documents),
        )

        self.vectorizer.fit(documents)

        self._is_fitted = True

        self.logger.info(
            "TF-IDF vectorizer fitted successfully with %d features.",
            len(self.vectorizer.vocabulary_),
        )

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        """
        Transform text using the fitted TF-IDF vectorizer.

        Args:
            texts: Email text documents to transform.

        Returns:
            csr_matrix: Sparse TF-IDF feature matrix.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
            TypeError: If texts is invalid.
            ValueError: If texts contains invalid documents.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "TF-IDF vectorizer must be fitted before transformation."
            )

        documents = self._validate_texts(texts)

        self.logger.info(
            "Transforming %d text documents into TF-IDF features.",
            len(documents),
        )

        return self.vectorizer.transform(documents)

    def fit_transform(
        self,
        training_texts: Iterable[str],
    ) -> csr_matrix:
        """
        Fit the vectorizer on training text and transform that text.

        This method is intended for the training dataset only.

        Args:
            training_texts: Email text documents from the training set.

        Returns:
            csr_matrix: Sparse TF-IDF training feature matrix.

        Raises:
            TypeError: If training_texts is invalid.
            ValueError: If training_texts contains invalid documents.
        """

        documents = self._validate_texts(training_texts)

        self.logger.info(
            "Fitting and transforming %d training documents.",
            len(documents),
        )

        features = self.vectorizer.fit_transform(documents)

        self._is_fitted = True

        self.logger.info(
            "TF-IDF feature extraction completed. Generated %d features.",
            features.shape[1],
        )

        return features

    def save(self, output_path: str | Path) -> None:
        """
        Save the fitted TF-IDF vectorizer to disk.

        Args:
            output_path: Destination path for the serialized vectorizer.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
            TypeError: If output_path is invalid.
            ValueError: If output_path is empty.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted TF-IDF vectorizer."
            )

        if not isinstance(output_path, (str, Path)):
            raise TypeError(
                "output_path must be a string or Path."
            )

        output_path = Path(output_path)

        if not str(output_path).strip():
            raise ValueError("output_path cannot be empty.")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.vectorizer,
            output_path,
        )

        self.logger.info(
            "TF-IDF vectorizer saved to: %s",
            output_path,
        )

    @classmethod
    def load(cls, input_path: str | Path) -> "TextFeatureExtractor":
        """
        Load a previously fitted TF-IDF vectorizer.

        Args:
            input_path: Path to the saved vectorizer artifact.

        Returns:
            TextFeatureExtractor: Extractor containing the fitted vectorizer.

        Raises:
            TypeError: If input_path is invalid.
            ValueError: If input_path is empty.
            FileNotFoundError: If the artifact does not exist.
        """

        if not isinstance(input_path, (str, Path)):
            raise TypeError(
                "input_path must be a string or Path."
            )

        input_path = Path(input_path)

        if not str(input_path).strip():
            raise ValueError("input_path cannot be empty.")

        if not input_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer artifact not found: {input_path}"
            )

        if not input_path.is_file():
            raise ValueError(
                f"TF-IDF vectorizer path is not a file: {input_path}"
            )

        extractor = cls()

        extractor.vectorizer = joblib.load(input_path)
        extractor._is_fitted = True

        extractor.logger.info(
            "TF-IDF vectorizer loaded from: %s",
            input_path,
        )

        return extractor