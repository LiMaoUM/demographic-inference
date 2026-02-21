"""
Example configuration file for demographic inference trainers.

This file contains example configurations for each training approach:
1. Causal Language Model (SFT) - Text generation
2. Sequence Classification - Direct demographic prediction
3. ModernBERT Single-task - Single attribute prediction
4. ModernBERT Multi-task - Multiple attributes simultaneously
"""

from src.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig

# Example 1: Causal Language Model - Twitter ID prediction
CLM_TWITTER_ID = Config(
    data=DataConfig(
        data_dir="data",
        platform="twitter",
        feature="id",
        version=1,
        number_of_atleast_posts=20,
        number_of_posts_per_sample=20,
        number_of_samples=8,
    ),
    model=ModelConfig(
        model_id="google/gemma-3-12b-it",
        max_seq_length=8000,
    ),
    training=TrainingConfig(
        output_dir="models/clm_twitter_id",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        bf16=True,
        warmup_steps=5,
        cuda_device=0,
    ),
    wandb=WandbConfig(
        enabled=True,
        project="demographic-inference-clm",
    ),
)

# Example 2: Sequence Classification - Both platforms, Party prediction
SEQ_CLS_BOTH_PARTY = Config(
    data=DataConfig(
        data_dir="data",
        platform="both",
        feature="party",
        version=1,
        number_of_atleast_posts=20,
        number_of_posts_per_sample=20,
        number_of_samples=8,
    ),
    model=ModelConfig(
        model_id="google/gemma-3-12b-it",
        max_seq_length=12800,
    ),
    training=TrainingConfig(
        output_dir="models/seq_cls_both_party",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        bf16=True,
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        cuda_device=0,
    ),
    wandb=WandbConfig(
        enabled=True,
        project="demographic-inference-seqcls",
    ),
)

# Example 3: ModernBERT Single-task - Twitter Age prediction
MODERNBERT_TWITTER_AGE = Config(
    data=DataConfig(
        data_dir="data",
        platform="twitter",
        feature="age",
        version=1,
        number_of_atleast_posts=20,
        number_of_posts_per_sample=20,
        number_of_samples=8,
    ),
    model=ModelConfig(
        model_id="answerdotai/ModernBERT-large",
        max_seq_length=8000,
    ),
    training=TrainingConfig(
        output_dir="models/modernbert_twitter_age",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=8e-5,
        bf16=False,
    ),
    wandb=WandbConfig(
        enabled=True,
        project="demographic-inference-modernbert",
    ),
)

# Example 4: ModernBERT Multi-task Learning - Multiple attributes
MODERNBERT_MULTITASK = Config(
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
        output_dir="models/modernbert_multitask",
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
        project="demographic-inference-multitask",
    ),
)


def print_examples():
    """Print all available configuration examples."""
    print("\n" + "=" * 70)
    print("Demographic Inference Configuration Examples")
    print("=" * 70)
    print("\nAvailable configurations:")
    print("  1. CLM_TWITTER_ID")
    print("     - Approach: Causal Language Model (SFT)")
    print("     - Model: Gemma 3 12B")
    print("     - Task: Twitter ID prediction via text generation")
    print()
    print("  2. SEQ_CLS_BOTH_PARTY")
    print("     - Approach: Sequence Classification")
    print("     - Model: Gemma 3 12B")
    print("     - Task: Political party prediction (both platforms)")
    print()
    print("  3. MODERNBERT_TWITTER_AGE")
    print("     - Approach: Single-task Sequence Classification")
    print("     - Model: ModernBERT Large")
    print("     - Task: Age prediction (Twitter)")
    print()
    print("  4. MODERNBERT_MULTITASK")
    print("     - Approach: Multi-task Learning")
    print("     - Model: ModernBERT Large")
    print("     - Task: Multiple demographic attributes simultaneously")
    print("=" * 70 + "\n")




if __name__ == "__main__":
    print_examples()
