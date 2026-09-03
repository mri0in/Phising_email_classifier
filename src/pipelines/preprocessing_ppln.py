# src/pipelines/preprocessing_ppln.py

"""
Preprocessing pipeline for the phishing email classification system.

This module coordinates the preprocessing workflow by using the
DataProcessor component.

Responsibilities:
    - Validate preprocessing inputs.
    - Invoke DataProcessor.
    - Save the processed dataset.
    - Return a standardized PipelineResult.

The pipeline does not implement individual data-cleaning operations.
Those responsibilities belong to DataProcessor.
"""

from pathlib import Path

from src.orchestration.base_ppln import BasePipeline, PipelineResult
from src.preprocessing.data_processor import DataProcessor


class PreprocessingPipeline(BasePipeline):
    """
    Pipeline responsible for preparing the raw phishing email dataset.

    Workflow:
        Raw CSV
            ↓
        DataProcessor
            ↓
        Processed CSV
    """

    PIPELINE_NAME = "preprocessing"

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """
        Initialize the preprocessing pipeline.

        Args:
            input_path: Path to the raw phishing email CSV file.
            output_path: Path where the processed dataset will be saved.

        Raises:
            TypeError: If input_path or output_path is not a valid path type.
            ValueError: If either path is empty.
        """

        if not isinstance(input_path, (str, Path)):
            raise TypeError("input_path must be a string or Path.")

        if not isinstance(output_path, (str, Path)):
            raise TypeError("output_path must be a string or Path.")

        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        if not str(self.input_path).strip():
            raise ValueError("input_path cannot be empty.")

        if not str(self.output_path).strip():
            raise ValueError("output_path cannot be empty.")

    @property
    def name(self) -> str:
        """
        Return the unique pipeline name.

        Returns:
            str: Pipeline identifier.
        """

        return self.PIPELINE_NAME

    def run(self) -> PipelineResult:
        """
        Execute the preprocessing workflow.

        Returns:
            PipelineResult: Standardized result describing the execution.

        Raises:
            FileNotFoundError: If the raw dataset does not exist.
            RuntimeError: If preprocessing fails.
        """

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {self.input_path}"
            )

        if not self.input_path.is_file():
            raise ValueError(
                f"Input path is not a file: {self.input_path}"
            )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        processor = DataProcessor(
            input_path=self.input_path,
            output_path=self.output_path,
        )

        try:
            processed_data = processor.process()

        except Exception as exc:
            raise RuntimeError(
                "Preprocessing pipeline failed."
            ) from exc

        return PipelineResult(
            pipeline_name=self.name,
            success=True,
            message="Preprocessing completed successfully.",
            metadata={
                "input_path": str(self.input_path),
                "output_path": str(self.output_path),
                "output_rows": len(processed_data),
            },
        )