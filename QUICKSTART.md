# Quick Start Guide

## 1. Installation (5 minutes)

```bash
# Clone/navigate to the project
cd demographic-inference

# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## 2. Prepare Your Data

Your data should be in one of these formats:

### Option A: Parquet Files
```
data/
├── twitter_age_train_20_20_8.parquet
├── twitter_age_test_20_20_8.parquet
```

Expected columns: `instruction`, `input_data`, `output_data`

### Option B: JSONL Files (for multi-task)
```
data/
├── twitter_seq_train.jsonl
├── twitter_seq_test.jsonl
```

Expected format:
```json
{
  "conversations": [
    {"role": "user", "content": "Tweet text..."},
    {"role": "assistant", "content": "{\"gender\": \"male\", \"age\": \"30-44\"}"}
  ]
}
```

## 3. Run Training (Choose One)

### ModernBERT (Recommended for beginners)
```bash
python train_modernbert.py
```
✅ Fastest training
✅ Good for single demographic attributes
✅ Low memory requirements

### Causal Language Model (for text generation)
```bash
python train_modernbert.py  # Use ModernBERTTrainer or CausalLanguageModelTrainer
```
✅ Better language understanding
✅ Good for free-form text generation
✅ Model configurable (Gemma, Llama, etc.)

### Sequence Classification (for direct prediction)
```bash
python train_modernbert.py  # Use SequenceClassificationTrainer
```
✅ Balanced performance
✅ Good for multi-class classification
✅ Model configurable via config

## 4. Customize Configuration

Edit the config in your training script:

```python
from src.config import Config, DataConfig, ModelConfig, TrainingConfig

config = Config(
    data=DataConfig(
        data_dir="data",
        platform="twitter",        # or "reddit", "both"
        feature="age",             # or "gender", "party", etc.
        number_of_atleast_posts=20,
        number_of_posts_per_sample=20,
        number_of_samples=8,
    ),
    model=ModelConfig(
        model_id="answerdotai/ModernBERT-large",
        max_seq_length=8000,
    ),
    training=TrainingConfig(
        output_dir="models",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=8e-5,
        bf16=False,  # Set to True if you have Ampere+ GPU
    ),
    wandb=WandbConfig(
        enabled=True,
        project="demographic-inference",
    ),
)
```

See `config_examples.py` for more pre-configured examples.

## 5. Monitor Training

### Option A: Weights & Biases (Online)
Set up W&B before training:
```bash
wandb login  # Follow prompts
```

Then enable in config:
```python
wandb=WandbConfig(enabled=True, project="your-project-name")
```

Visit https://wandb.ai to see real-time metrics

### Option B: Local Logs
Training logs are saved to:
```
models/training.log
```

## 6. Use Trained Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load trained model
model_path = "models/ModernBERT-large-twitter-age-20-20-8-v1"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Make predictions
text = "This is a sample tweet about my daily activities..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)
```

## 7. Advanced: Multi-task Learning

Train on multiple demographic attributes simultaneously:

```python
from src.training import ModernBERTTrainer

trainer = ModernBERTTrainer(config)
trainer.train_multi_task(
    train_path="data/twitter_seq_train.jsonl",
    val_path="data/twitter_seq_test.jsonl",
    label_fields=['gender', 'age', 'party_identification', 'income'],
    label_vocab={
        'gender': ['male', 'female'],
        'age': ['18-29', '30-44', '45-59', '60+'],
        'party_identification': ['Democrat', 'Republican', 'Independent', 'Other'],
        'income': ['Low income', 'Middle income', 'High income'],
    },
    dataset_from='twitter',
)
```

## 8. Project Structure Reference

```
demographic-inference/
├── train_*.py              # Entry point scripts
├── config_examples.py      # Configuration examples
├── src/
│   ├── config/            # Configuration classes
│   ├── models/            # Model implementations
│   ├── training/          # Trainer classes
│   └── utils/             # Utility functions
├── pyproject.toml         # Dependencies
├── README.md              # Full documentation
├── MIGRATION.md           # Refactoring details
└── QUICKSTART.md          # This file
```

## Troubleshooting

### "CUDA out of memory"
→ Reduce `per_device_train_batch_size` (e.g., 4 or 2)

### "FileNotFoundError: data file not found"
→ Check `data_dir` in config matches actual data location

### "ModuleNotFoundError: No module named 'src'"
→ Make sure you're running from project root:
```bash
cd /path/to/demographic-inference
python train_modernbert.py
```

### "W&B login required"
→ Either run `wandb login` or set `wandb.enabled=False` in config

### Slow training
→ Enable `bf16=True` if GPU supports it (e.g., A100, RTX 3090)

## Next Steps

1. ✅ Set up environment (`uv sync`)
2. ✅ Prepare data (parquet or JSONL)
3. ✅ Run training (`python train_*.py`)
4. ✅ Monitor progress (W&B or logs)
5. ✅ Evaluate results
6. ✅ Use trained model for predictions

## Resources

- **Full Documentation**: See `README.md`
- **Configuration Reference**: See `src/config/config.py`
- **Configuration Examples**: See `config_examples.py`
- **Migration from Old Code**: See `MIGRATION.md`

## Need Help?

- Check `README.md` for detailed documentation
- Review `config_examples.py` for working configurations
- See `src/training/*.py` for trainer implementations
- Check training logs in `models/training.log`

Good luck! 🚀
