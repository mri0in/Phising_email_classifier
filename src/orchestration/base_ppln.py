#src/orchestration/base_ppln.py
"""
Base pipeline contract and standardized pipeline execution result.

This module defines the common interface that every pipeline in the
phishing email classification system must follow.

Architecture:
    PipelineOrchestrator
            |
            v
       BasePipeline
            |
            +-- PreprocessingPipeline
            +-- FeaturePipeline
            +-- TrainingPipeline
            +-- InferencePipeline
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineResult:
    """
    Standardized result returned by every pipeline execution.

    Attributes:
        pipeline_name: Name of the pipeline that was executed.
        success: Whether the pipeline completed successfully.
        message: Human-readable execution status.
        metadata: Additional information produced during execution.
    """

    pipeline_name: str
    success: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the pipeline result after initialization."""

        if not isinstance(self.pipeline_name, str):
            raise TypeError("pipeline_name must be a string.")

        if not self.pipeline_name.strip():
            raise ValueError("pipeline_name cannot be empty.")

        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean.")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")


class BasePipeline(ABC):
    """
    Abstract base class for all project pipelines.

    Every concrete pipeline must implement the `run()` method.

    The orchestrator depends on this contract rather than depending on
    the implementation details of individual pipelines.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the pipeline.

        Returns:
            str: Pipeline identifier used by the orchestrator.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> PipelineResult:
        """
        Execute the pipeline workflow.

        Returns:
            PipelineResult: Standardized result of pipeline execution.
        """
        raise NotImplementedError