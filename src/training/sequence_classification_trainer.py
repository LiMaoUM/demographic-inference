"""Sequence classification trainer for demographic inference."""
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import pandas as pd

from ..config import Config


logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """Custom trainer with custom loss computation."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with custom handling for imbalanced data.

        Args:
            model: The model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return outputs

        Returns:
            Loss value or tuple of (loss, outputs)
        """
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


class SequenceClassificationTrainer:
    """Trainer for sequence classification tasks (e.g., demographic prediction).

    Uses Gemma 12B model with LoRA fine-tuning and 8-bit quantization.
    """

    def __init__(self, config: Config):
        """Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self._setup_logging()
        self._setup_cuda()
        self._initialize_wandb()

    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.config.training.output_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def _setup_cuda(self):
        """Setup CUDA devices."""
        if self.config.training.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.config.training.cuda_device
            )

    def _initialize_wandb(self):
        """Initialize Weights & Biases."""
        if self.config.wandb.enabled:
            model_name = self.config.model.get_model_name(self.config.data)
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.run_name or model_name,
            )

    def train(self):
        """Run training for sequence classification."""
        # Load datasets
        train_df = pd.read_parquet(self.config.data.train_file)
        val_df = pd.read_parquet(self.config.data.test_file)

        # Prepare data
        train_df.rename(
            columns={"output_data": "label", "input_data": "text"}, inplace=True
        )
        train_df["label"] = train_df["label"].astype("category")
        train_df["target"] = train_df["label"].cat.codes

        val_df.rename(
            columns={"output_data": "label", "input_data": "text"}, inplace=True
        )
        val_df["label"] = val_df["label"].astype("category")
        val_df["target"] = val_df["label"].cat.codes

        # Create datasets
        dataset_train = Dataset.from_pandas(train_df.drop("label", axis=1))
        dataset_val = Dataset.from_pandas(val_df.drop("label", axis=1))
        dataset_train_shuffled = dataset_train.shuffle(seed=42)

        dataset = DatasetDict({
            "train": dataset_train_shuffled,
            "validation": dataset_val,
        })

        # Load model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.model_id,
            quantization_config=quantization_config,
            num_labels=len(train_df["label"].cat.categories),
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id, add_prefix_space=True
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pretraining_tp = 1

        # Preprocess
        def preprocess_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=8192)

        tokenized_datasets = dataset.map(
            preprocess_fn, batched=True, remove_columns=["instruction", "text"]
        )
        tokenized_datasets = tokenized_datasets.rename_column("target", "label")
        tokenized_datasets.set_format("torch")

        # Training
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy_score(predictions, labels),
                "f1": f1_score(predictions, labels, average="macro"),
            }

        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            learning_rate=self.config.training.learning_rate,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,
            num_train_epochs=self.config.training.num_train_epochs,
            weight_decay=self.config.training.weight_decay,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_steps=10,
            save_steps=50,
            eval_steps=50,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.evaluate()

        # Save model
        model_name = self.config.model.get_model_name(self.config.data)
        model_path = os.path.join(self.config.training.output_dir, model_name)
        trainer.save_model(model_path)

        logger.info(f"Model saved to {model_path}")

        if self.config.wandb.enabled:
            wandb.finish()
