# src/main.py

import logging

from src.orchestration.pipeline_orchestrator import PipelineOrchestrator
from src.pipelines.feature_ppln import FeaturePipeline
from src.pipelines.preprocessing_ppln import PreprocessingPipeline
from src.pipelines.training_ppln import TrainingPipeline
from src.pipelines.inference_ppln import InferencePipeline
from src.pipelines.evaluation_ppln import EvaluationPipeline


def configure_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Application entry point."""

    configure_logging()

    preprocessing_pipeline = PreprocessingPipeline(
        input_path="data/raw/Phishing_Email.csv",
        output_path="data/processed/phishing_emails_processed.csv",
    )

    feature_pipeline = FeaturePipeline(
        input_path="data/processed/phishing_emails_processed.csv",
        output_dir="artifacts/features",
        vectorizer_path="artifacts/models/tfidf_vectorizer.joblib",
        random_state=42,
    )

    training_pipeline = TrainingPipeline(
        features_dir="artifacts/features",
        model_output_path="artifacts/models/phishing_classifier.joblib",
        random_state=42,
    )

    inference_pipeline = InferencePipeline(
        vectorizer_path="artifacts/models/tfidf_vectorizer.joblib",
        model_path="artifacts/models/phishing_classifier.joblib",
        email_text=(
            "URGENT: Your account has been suspended. "
            "Click the link immediately to verify your account "
            "and prevent permanent closure."
        ),
    )   

    evaluation_pipeline = EvaluationPipeline(
    features_dir="artifacts/features",
    model_path="artifacts/models/phishing_classifier.joblib",
    metrics_output_path="artifacts/metrics/evaluation_metrics.joblib",
)

    orchestrator = PipelineOrchestrator(
        pipelines={
            preprocessing_pipeline.name: preprocessing_pipeline,
            feature_pipeline.name: feature_pipeline,
            training_pipeline.name: training_pipeline,
            inference_pipeline.name: inference_pipeline,
            evaluation_pipeline.name: evaluation_pipeline,
        }
    )
    ######result = orchestrator.run("preprocessing")
    ####result = orchestrator.run("features")
    ###result = orchestrator.run("training")
    ###result = orchestrator.run("inference")
    result = orchestrator.run("evaluation")


    print("\nPipeline Result")
    print("----------------")
    print(f"Pipeline : {result.pipeline_name}")
    print(f"Success  : {result.success}")
    print(f"Message  : {result.message}")
    print(f"Metadata : {result.metadata}")


if __name__ == "__main__":
    main()