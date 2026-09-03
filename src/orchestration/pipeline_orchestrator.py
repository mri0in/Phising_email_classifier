# src/orchestration/pipeline_orchestrator.py

"""
Pipeline orchestration for the phishing email classification system.

This module is responsible for registering, selecting, and executing
project pipelines.

Architecture:
    main.py
        |
        v
    PipelineOrchestrator
        |
        +-- PreprocessingPipeline
        +-- FeaturePipeline
        +-- TrainingPipeline
        +-- InferencePipeline
"""

import logging

from src.orchestration.base_ppln import BasePipeline, PipelineResult


class PipelineOrchestrator:
    """
    Central controller responsible for executing registered pipelines.

    The orchestrator does not implement the business logic of any
    individual pipeline. It only manages pipeline registration,
    selection, and execution.
    """

    def __init__(
        self,
        pipelines: dict[str, BasePipeline],
    ) -> None:
        """
        Initialize the pipeline orchestrator.

        Args:
            pipelines: Dictionary mapping pipeline names to pipeline
                instances.

        Raises:
            TypeError: If pipelines is not a dictionary or contains
                invalid pipeline objects.
            ValueError: If a pipeline name is empty or duplicated.
        """

        if not isinstance(pipelines, dict):
            raise TypeError("pipelines must be a dictionary.")

        self._logger = logging.getLogger(self.__class__.__name__)
        self._pipelines: dict[str, BasePipeline] = {}

        for pipeline_name, pipeline in pipelines.items():
            self._register_pipeline(pipeline_name, pipeline)

    def _register_pipeline(
        self,
        pipeline_name: str,
        pipeline: BasePipeline,
    ) -> None:
        """
        Register a single pipeline with the orchestrator.

        Args:
            pipeline_name: Unique name used to identify the pipeline.
            pipeline: Pipeline instance implementing BasePipeline.

        Raises:
            TypeError: If the name or pipeline type is invalid.
            ValueError: If the pipeline name is empty or already
                registered.
        """

        if not isinstance(pipeline_name, str):
            raise TypeError("Pipeline name must be a string.")

        if not pipeline_name.strip():
            raise ValueError("Pipeline name cannot be empty.")

        if not isinstance(pipeline, BasePipeline):
            raise TypeError(
                f"Pipeline '{pipeline_name}' must inherit from "
                "BasePipeline."
            )

        normalized_name = pipeline_name.strip().lower()

        if normalized_name in self._pipelines:
            raise ValueError(
                f"Pipeline '{normalized_name}' is already registered."
            )

        if pipeline.name.strip().lower() != normalized_name:
            raise ValueError(
                f"Pipeline name mismatch: registry name "
                f"'{normalized_name}' does not match pipeline name "
                f"'{pipeline.name}'."
            )

        self._pipelines[normalized_name] = pipeline

        self._logger.debug(
            "Registered pipeline: %s",
            normalized_name,
        )

    def run(self, pipeline_name: str) -> PipelineResult:
        """
        Execute a registered pipeline.

        Args:
            pipeline_name: Name of the pipeline to execute.

        Returns:
            PipelineResult: Result returned by the selected pipeline.

        Raises:
            TypeError: If pipeline_name is not a string.
            ValueError: If pipeline_name is empty.
            KeyError: If the requested pipeline is not registered.
        """

        if not isinstance(pipeline_name, str):
            raise TypeError("pipeline_name must be a string.")

        normalized_name = pipeline_name.strip().lower()

        if not normalized_name:
            raise ValueError("pipeline_name cannot be empty.")

        if normalized_name not in self._pipelines:
            available_pipelines = ", ".join(
                sorted(self._pipelines.keys())
            )

            raise KeyError(
                f"Pipeline '{normalized_name}' is not registered. "
                f"Available pipelines: {available_pipelines}"
            )

        pipeline = self._pipelines[normalized_name]

        self._logger.info(
            "Starting pipeline: %s",
            normalized_name,
        )

        try:
            result = pipeline.run()

        except Exception:
            self._logger.exception(
                "Pipeline failed: %s",
                normalized_name,
            )
            raise

        if not isinstance(result, PipelineResult):
            raise TypeError(
                f"Pipeline '{normalized_name}' returned an invalid result. "
                "Expected PipelineResult."
            )

        self._logger.info(
            "Pipeline completed: %s | success=%s",
            normalized_name,
            result.success,
        )

        return result

    def list_pipelines(self) -> list[str]:
        """
        Return the names of all registered pipelines.

        Returns:
            list[str]: Sorted list of registered pipeline names.
        """

        return sorted(self._pipelines.keys())