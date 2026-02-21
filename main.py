"""
Quick start example for demographic inference.

This script demonstrates how to use the different trainers
for demographic attribute prediction.
"""

from src.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig
from src.training import (
    CausalLanguageModelTrainer,
    SequenceClassificationTrainer,
    ModernBERTTrainer,
)


def example_causal_lm():
    """Example: Causal Language Model training for text generation."""
    print("\n=== Causal Language Model Example ===")
    print("Use case: Text generation with SFT")
    print("Model: Gemma 3 12B")
    print("Task: Demographic attribute prediction via generation")

    config = Config(
        data=DataConfig(
            data_dir="data",
            platform="twitter",
            feature="id",
            version=1,
        ),
        model=ModelConfig(
            model_id="google/gemma-3-12b-it",
            max_seq_length=8000,
        ),
        training=TrainingConfig(
            output_dir="models/clm",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            learning_rate=5e-4,
            bf16=True,
        ),
        wandb=WandbConfig(
            enabled=True,
            project="demographic-inference",
        ),
    )

    trainer = CausalLanguageModelTrainer(config)
    print("To train, uncomment: trainer.train()")


def example_sequence_classification():
    """Example: Sequence Classification for direct demographic prediction."""
    print("\n=== Sequence Classification Example ===")
    print("Use case: Multi-class demographic classification")
    print("Model: Gemma 3 12B")
    print("Task: Direct demographic attribute prediction")

    config = Config(
        data=DataConfig(
            data_dir="data",
            platform="both",
            feature="party",
            version=1,
        ),
        model=ModelConfig(
            model_id="google/gemma-3-12b-it",
            max_seq_length=12800,
        ),
        training=TrainingConfig(
            output_dir="models/seq_cls",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            bf16=True,
            eval_steps=50,
            save_steps=50,
            logging_steps=10,
        ),
        wandb=WandbConfig(
            enabled=True,
            project="demographic-inference",
        ),
    )

    trainer = SequenceClassificationTrainer(config)
    print("To train, uncomment: trainer.train()")


def example_modernbert_single_task():
    """Example: ModernBERT single-task training."""
    print("\n=== ModernBERT Single-task Example ===")
    print("Use case: Single demographic attribute prediction")
    print("Model: ModernBERT Large")
    print("Task: Age prediction")

    config = Config(
        data=DataConfig(
            data_dir="data",
            platform="twitter",
            feature="age",
            version=1,
        ),
        model=ModelConfig(
            model_id="answerdotai/ModernBERT-large",
            max_seq_length=8000,
        ),
        training=TrainingConfig(
            output_dir="models/modernbert_single",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=8e-5,
            bf16=False,
        ),
        wandb=WandbConfig(
            enabled=True,
            project="demographic-inference",
        ),
    )

    trainer = ModernBERTTrainer(config)
    print("To train, uncomment: trainer.train_single_task()")


def example_modernbert_multi_task():
    """Example: ModernBERT multi-task training."""
    print("\n=== ModernBERT Multi-task Example ===")
    print("Use case: Multiple demographic attributes simultaneously")
    print("Model: ModernBERT Large")
    print("Tasks: Gender, Age, Party prediction")

    config = Config(
        data=DataConfig(
            data_dir="data",
            platform="twitter",
            feature="multi",
            version=1,
        ),
        model=ModelConfig(
            model_id="answerdotai/ModernBERT-large",
            max_seq_length=8000,
        ),
        training=TrainingConfig(
            output_dir="models/modernbert_multi",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            bf16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
        ),
        wandb=WandbConfig(
            enabled=True,
            project="demographic-inference",
        ),
    )

    trainer = ModernBERTTrainer(config)

    label_fields = ["gender", "age", "party"]
    label_vocab = {
        "gender": ["male", "female"],
        "age": ["18-29", "30-44", "45-59", "60+"],
        "party": ["Democrat", "Republican", "Independent", "Other"],
    }

    print("To train, uncomment:")
    print("trainer.train_multi_task(...)")


def main():
    """Display all available examples."""
    print("\n" + "=" * 60)
    print("Demographic Inference - Quick Start Examples")
    print("=" * 60)

    example_causal_lm()
    example_sequence_classification()
    example_modernbert_single_task()
    example_modernbert_multi_task()

    print("\n" + "=" * 60)
    print("Available Training Approaches:")
    print("-" * 60)
    print("1. CausalLanguageModelTrainer: Text generation (SFT)")
    print("2. SequenceClassificationTrainer: Direct classification")
    print("3. ModernBERTTrainer: Single or multi-task learning")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
