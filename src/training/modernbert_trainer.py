"""Training module for ModernBERT models."""
import ast
import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from datasets import (
    load_dataset,
    DatasetDict,
    ClassLabel,
    concatenate_datasets,
)
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from ..config import Config
from ..models import ModernBERTClassifier, ModernBERTMultiTaskClassifier


logger = logging.getLogger(__name__)


class MetricsCallback(TrainerCallback):
    """Callback to track training metrics."""

    def __init__(self):
        """Initialize callback."""
        self.training_history = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics.

        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Logs dictionary
            kwargs: Additional arguments
        """
        if logs is not None:
            if "loss" in logs:
                self.training_history["train"].append(logs)
            elif "eval_loss" in logs:
                self.training_history["eval"].append(logs)


class ModernBERTTrainer:
    """Trainer for ModernBERT models."""

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

    def train_single_task(self):
        """Train single-task classifier."""
        # Load datasets
        train_df = pd.read_parquet(self.config.data.train_file)
        val_df = pd.read_parquet(self.config.data.test_file)

        train_df.rename(
            columns={"input_data": "text", "output_data": "label"}, inplace=True
        )
        train_df.drop(columns=["instruction"], inplace=True)

        val_df.rename(
            columns={"input_data": "text", "output_data": "label"}, inplace=True
        )
        val_df.drop(columns=["instruction"], inplace=True)

        from datasets import Dataset

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

        label_names = sorted(train_df["label"].unique().tolist())
        dataset["train"] = dataset["train"].cast_column(
            "label", ClassLabel(names=label_names)
        )
        dataset["validation"] = dataset["validation"].cast_column(
            "label", ClassLabel(names=label_names)
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id, reference_compile=False
        )

        def preprocess_fn(examples):
            return tokenizer(examples["text"], truncation=True)

        tokenized_datasets = dataset.map(preprocess_fn, batched=True)

        # Compute metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            metrics_dict = {
                "f1_score": f1_score(
                    np.argmax(predictions, axis=-1), labels, average="macro"
                ),
                "accuracy": accuracy_score(
                    np.argmax(predictions, axis=-1), labels
                ),
            }
            return metrics_dict

        # Load model
        model = ModernBERTClassifier(
            self.config.model.model_id,
            num_labels=len(label_names),
        )

        # Data collator
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            learning_rate=self.config.training.learning_rate,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            num_train_epochs=self.config.training.num_train_epochs,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            bf16=self.config.training.bf16,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        metrics_callback = MetricsCallback()
        trainer.add_callback(metrics_callback)
        trainer.train()

        logger.info("Single-task training completed")

    def train_multi_task(
        self,
        train_path: str,
        val_path: str,
        label_fields: List[str],
        label_vocab: Dict[str, List[str]],
        extract_fn: Optional[Callable] = None,
        dataset_from: str = "twitter",
    ):
        """Train multi-task classifier.

        Args:
            train_path: Path to training JSONL file
            val_path: Path to validation JSONL file
            label_fields: List of label field names
            label_vocab: Dictionary mapping field names to label names
            extract_fn: Function to extract fields from examples
            dataset_from: Dataset source ('twitter' or 'reddit')
        """
        if extract_fn is None:
            extract_fn = self._default_extract_fn(label_fields, dataset_from)

        # Load datasets
        dataset = DatasetDict({
            "train": load_dataset("json", data_files=train_path, split="train"),
            "validation": load_dataset("json", data_files=val_path, split="train"),
        })

        for split in ["train", "validation"]:
            dataset[split] = dataset[split].map(extract_fn)

        # Encode labels
        for field in label_fields:
            dataset["train"] = dataset["train"].cast_column(
                field, ClassLabel(names=label_vocab[field])
            )
            dataset["validation"] = dataset["validation"].cast_column(
                field, ClassLabel(names=label_vocab[field])
            )

        # Tokenization
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_id)

        def preprocess_fn(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)

        dataset = dataset.map(preprocess_fn, batched=True)

        # Model
        label_dims = {task: len(label_vocab[task]) for task in label_fields}
        model = ModernBERTMultiTaskClassifier(
            self.config.model.model_id, label_dims
        )

        # Metrics
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            metrics = {}
            for task in logits:
                preds = torch.tensor(logits[task]).argmax(dim=1).numpy()
                true = labels[task]
                metrics[f"{task}_acc"] = accuracy_score(true, preds)
                metrics[f"{task}_f1"] = f1_score(true, preds, average="macro")
            return metrics

        # Wrap labels
        def wrap_labels(example):
            example["labels"] = {task: example[task] for task in label_fields}
            return example

        def custom_collate_fn(features):
            label_dicts = [f.pop("labels") for f in features]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            batch["labels"] = {
                field: torch.tensor(
                    [d[field] for d in label_dicts], dtype=torch.long
                )
                for field in label_dicts[0]
            }
            return batch

        dataset = dataset.map(wrap_labels)

        # Training
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.training.learning_rate,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            num_train_epochs=self.config.training.num_train_epochs,
            weight_decay=self.config.training.weight_decay,
            logging_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=custom_collate_fn,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        logger.info("Multi-task training completed")

        if self.config.wandb.enabled:
            wandb.finish()

    @staticmethod
    def _default_extract_fn(label_fields: List[str], dataset_from: str):
        """Create default extraction function.

        Args:
            label_fields: List of label field names
            dataset_from: Dataset source

        Returns:
            Extraction function
        """

        def extract_fields(example):
            conversations = example.get("conversations", [])
            if len(conversations) < 2:
                return {
                    "text": "",
                    **{field: "Unknown" for field in label_fields},
                }

            if dataset_from == "twitter":
                user_msg = conversations[0].get("content", "")
                assistant_msg = conversations[1].get("content", "{}")
            else:
                user_msg = conversations[1].get("content", "")
                assistant_msg = conversations[2].get("content", "{}")

            result = {"text": user_msg}

            try:
                if dataset_from == "twitter":
                    parsed_labels = json.loads(assistant_msg)
                else:
                    parsed_labels = ast.literal_eval(assistant_msg)

                for field in label_fields:
                    val = parsed_labels.get(field, "Unknown")
                    if val is None:
                        val = "Unknown"
                    result[field] = str(val)
            except Exception as e:
                logger.warning(f"Failed to parse: {e}")
                for field in label_fields:
                    result[field] = "Unknown"

            return result

        return extract_fields
