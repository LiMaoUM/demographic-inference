# Demographic Inference

A comprehensive framework for demographic attribute inference using language models with support for multiple training approaches: **Causal Language Modeling** (SFT), **Sequence Classification**, and **Multi-task Learning**.

## Project Structure

```
demographic-inference/
├── src/
│   ├── __init__.py
│   ├── config/                                  # Configuration management
│   │   ├── __init__.py
│   │   └── config.py                            # Config classes
│   ├── models/                                  # Model implementations
│   │   ├── __init__.py
│   │   └── modern_bert.py                       # ModernBERT classifier models
│   ├── training/                                # Training modules
│   │   ├── __init__.py
│   │   ├── causal_language_model_trainer.py     # SFT approach
│   │   ├── sequence_classification_trainer.py   # Classification approach
│   │   └── modernbert_trainer.py                # Multi-task approach
│   └── utils/                                   # Utility functions
│       ├── __init__.py
│       ├── data_utils.py                        # Data loading and preprocessing
│       └── metrics.py                           # Metrics computation
├── main.py                                      # Quick start example
├── config_examples.py                           # Configuration examples
├── pyproject.toml                               # Project configuration and dependencies
├── README.md                                    # This file
└── .venv/                                       # Virtual environment (created by uv)
```

## Setup & Installation

### Using UV (Recommended)

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

### Manual Installation with pip

```bash
pip install -r requirements.txt
```

## Configuration

The project uses a configuration system with the following components:

- **DataConfig**: Data loading and dataset configuration
  - `data_dir`: Directory containing data files
  - `platform`: Platform source (e.g., 'twitter', 'reddit', 'both')
  - `feature`: Target demographic feature
  - `version`: Configuration version

- **ModelConfig**: Model selection and settings
  - `model_id`: HuggingFace model identifier
  - `max_seq_length`: Maximum sequence length

- **TrainingConfig**: Training hyperparameters
  - `learning_rate`: Learning rate (default: 5e-5)
  - `num_train_epochs`: Number of training epochs
  - `per_device_train_batch_size`: Training batch size
  - `bf16`: Use bfloat16 precision (default: True)

- **WandbConfig**: Weights & Biases settings
  - `enabled`: Enable W&B logging
  - `project`: W&B project name

## Training Approaches

This project supports three distinct training methods. Each method can work with any compatible model via configuration.

### 1. Causal Language Model Trainer (SFT)

**Use case:** Text generation and completion tasks  
**Default Model:** Gemma 3 12B Instruct (configurable)
**Features:**
- Supervised Fine-Tuning (SFT) with LoRA adapters
- 8-bit quantization for memory efficiency
- Prompt formatting: `Instruction: {instruction}\nInput: {input}\nOutput: {output}`
- Metrics: Perplexity, token accuracy
- Best for: Free-form demographic inference with natural language generation

**Example:**
```python
from src.training import CausalLanguageModelTrainer
from src.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig

# Configure your desired model in the ModelConfig
config = Config(
    data=DataConfig(data_dir="data", platform="twitter", feature="id"),
    model=ModelConfig(
        model_id="google/gemma-3-12b-it",  # Change this to any compatible model
        max_seq_length=8000
    ),
    training=TrainingConfig(
        output_dir="models/clm",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        bf16=True,
    ),
    wandb=WandbConfig(enabled=True, project="demographic-inference"),
)

trainer = CausalLanguageModelTrainer(config)
trainer.train()  # Train using the Causal Language Model method
```

### 2. Sequence Classification Trainer

**Use case:** Multi-class classification of demographic attributes  
**Model:** Gemma 3 12B Instruct  
**Features:**
- Multi-class classification with custom loss function for imbalanced data
- 8-bit quantization with LoRA adapters
- Custom cross-entropy loss weighting
- Metrics: Accuracy, Precision, Recall, F1 (weighted and macro)
- Best for: Direct demographic attribute prediction

**Example:**
```python
from src.training import SequenceClassificationTrainer
from src.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig

# Configure your desired model in the ModelConfig
config = Config(
    data=DataConfig(data_dir="data", platform="twitter", feature="party"),
    model=ModelConfig(
        model_id="google/gemma-3-12b-it",  # Change this to any compatible model
        max_seq_length=12800
    ),
    training=TrainingConfig(
        output_dir="models/seq_cls",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        bf16=True,
    ),
    wandb=WandbConfig(enabled=True, project="demographic-inference"),
)

trainer = SequenceClassificationTrainer(config)
trainer.train()  # Train using the Sequence Classification method
```

### 3. ModernBERT Trainer (Single & Multi-task)

**Use case:** Single-task and multi-task demographic prediction  
**Model:** ModernBERT Large  
**Features:**
- Single-task sequence classification
- Multi-task learning for multiple demographic attributes simultaneously
- Automatic label vocabulary creation
- Epoch-based evaluation
- Best for: Comprehensive multi-attribute demographic inference

**Single-task Example:**
```python
from src.training import ModernBERTTrainer
from src.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig

config = Config(
    data=DataConfig(data_dir="data", platform="twitter", feature="age"),
    model=ModelConfig(
        model_id="answerdotai/ModernBERT-large",  # Change this to any compatible model
        max_seq_length=8000
    ),
    training=TrainingConfig(
        output_dir="models/modernbert",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=8e-5,
    ),
    wandb=WandbConfig(enabled=True, project="demographic-inference"),
)

trainer = ModernBERTTrainer(config)
trainer.train_single_task()  # Train using the ModernBERT single-task method
```

**Multi-task Example:**
```python
label_fields = ['gender', 'age', 'party']
label_vocab = {
    'gender': ['male', 'female'],
    'age': ['18-29', '30-44', '45-59', '60+'],
    'party': ['Democrat', 'Republican', 'Independent', 'Other'],
}

trainer.train_multi_task(
    train_path="data/twitter_train.jsonl",
    val_path="data/twitter_val.jsonl",
    label_fields=label_fields,
    label_vocab=label_vocab,
)  # Train using the ModernBERT multi-task method
```

## Data Format

### Parquet Format (for Causal Language Model and Sequence Classification trainers)

Required columns:
- `instruction`: Task instruction
- `input_data`: Input text
- `output_data`: Target label

### JSONL Format (for ModernBERT multi-task)

Each line should contain:
```json
{
  "conversations": [
    {"role": "user", "content": "Text..."},
    {"role": "assistant", "content": "{\"gender\": \"male\", \"age\": \"30-44\", ...}"}
  ]
}
```

## Models Supported

The framework is model-agnostic. Each trainer can work with any compatible model via configuration:

### Recommended Models

#### Gemma 3 12B Instruct
- **Model ID:** `google/gemma-3-12b-it`
- **Works with:** CausalLanguageModelTrainer, SequenceClassificationTrainer
- **Max Sequence Length:** 12,800 tokens
- **Optimization:** 8-bit quantization + LoRA fine-tuning

#### ModernBERT Large
- **Model ID:** `answerdotai/ModernBERT-large`
- **Works with:** ModernBERTTrainer
- **Max Sequence Length:** 8,000 tokens
- **Features:** Modern architecture optimized for long sequences

### Using Different Models

You can easily swap models by changing the `model_id` in your configuration:

```python
config = Config(
    data=DataConfig(...),
    model=ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",  # Or any other compatible model
        max_seq_length=8000
    ),
    training=TrainingConfig(...),
    wandb=WandbConfig(...),
)
```

## Key Features

✅ **Three Distinct Training Methods**
  - **Causal Language Model (SFT):** Text generation approach for demographic inference
  - **Sequence Classification:** Direct classification approach for demographic prediction
  - **Multi-task Learning:** Predict multiple demographic attributes simultaneously

✅ **Model-Agnostic Design**
  - Use any compatible HuggingFace model with each trainer
  - Easy model swapping via configuration
  - Examples provided with recommended models (Gemma 3 12B, ModernBERT Large)

✅ **Efficient Training**
  - 8-bit quantization for memory efficiency
  - LoRA fine-tuning to reduce trainable parameters
  - Warmup and gradient accumulation support

✅ **Flexible Configuration System**
  - Dataclass-based configuration management
  - No hardcoded paths or parameters
  - Easy to create and manage multiple experiments

✅ **Multi-task Learning**
  - Train on multiple demographic attributes simultaneously
  - Automatic label vocabulary creation
  - Epoch-based evaluation and checkpointing

✅ **Production Ready**
  - Well-organized, modular codebase
  - Comprehensive error handling
  - Clear separation of concerns

✅ **Experiment Tracking**
  - Integrated Weights & Biases logging
  - Real-time training metrics
  - Easy experiment comparison

## Utilities

### Data Loading
- `load_parquet_datasets()`: Load from parquet files
- `load_jsonl_datasets()`: Load from JSONL files
- `preprocess_function()`: Tokenization and preprocessing

### Metrics
- `compute_metrics()`: Compute F1, accuracy
- `ComputeMetrics`: Class-based metrics computation

## Logging

All training logs are saved to:
- `models/training.log`: Training process logs
- W&B Dashboard: Real-time training metrics (if enabled)

## Dependencies

Core dependencies (see `pyproject.toml` for complete list):
- `torch>=2.0.0`: Deep learning framework
- `transformers>=4.30.0`: HuggingFace models
- `datasets>=2.10.0`: Dataset utilities
- `peft>=0.4.0`: Parameter-efficient fine-tuning
- `trl>=0.4.0`: Transformer reinforcement learning
- `wandb>=0.14.0`: Experiment tracking
- `scikit-learn>=1.2.0`: Metrics and utilities

## License

[Specify your license here]

## Contributing

[Contribution guidelines]

## Citation

[Citation information if applicable]
