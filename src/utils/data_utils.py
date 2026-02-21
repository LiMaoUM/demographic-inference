"""Data utilities for loading and preprocessing datasets."""
import os
from functools import partial
from typing import Dict, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, load_dataset


def load_parquet_datasets(
    train_file: str, test_file: str, rename_columns: Optional[Dict[str, str]] = None
) -> DatasetDict:
    """Load datasets from parquet files.

    Args:
        train_file: Path to training parquet file
        test_file: Path to test parquet file
        rename_columns: Dictionary of column renames

    Returns:
        DatasetDict with train and validation splits
    """
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(test_file)

    if rename_columns:
        train_df.rename(columns=rename_columns, inplace=True)
        val_df.rename(columns=rename_columns, inplace=True)

    # Drop unnecessary columns
    cols_to_drop = [col for col in train_df.columns if col == "instruction"]
    if cols_to_drop:
        train_df.drop(columns=cols_to_drop, inplace=True)
        val_df.drop(columns=cols_to_drop, inplace=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def load_jsonl_datasets(
    train_file: str,
    test_file: str,
    extract_fn,
    label_fields: Optional[list] = None,
) -> DatasetDict:
    """Load datasets from JSONL files.

    Args:
        train_file: Path to training JSONL file
        test_file: Path to test JSONL file
        extract_fn: Function to extract fields from examples
        label_fields: List of label field names

    Returns:
        DatasetDict with train and validation splits
    """
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    dataset = DatasetDict({
        "train": load_dataset("json", data_files=train_file, split="train"),
        "validation": load_dataset("json", data_files=test_file, split="train"),
    })

    for split in ["train", "validation"]:
        dataset[split] = dataset[split].map(extract_fn)

    return dataset


def preprocess_function(
    examples, tokenizer, task_inputs: list = None, max_length: int = 512
):
    """Preprocess examples for model training.

    Args:
        examples: Input examples
        tokenizer: HuggingFace tokenizer
        task_inputs: List of input column names
        max_length: Maximum sequence length

    Returns:
        Tokenized examples
    """
    if task_inputs is None:
        task_inputs = ["text"]

    inps = [examples[inp] for inp in task_inputs]
    return tokenizer(
        *inps, truncation=True, max_length=max_length, padding=True
    )


def create_label_maps(
    dataset: DatasetDict, train_split_name: str = "train"
) -> tuple:
    """Create label maps from dataset.

    Args:
        dataset: HuggingFace dataset
        train_split_name: Name of training split

    Returns:
        Tuple of (id2label, label2id) dicts
    """
    labels = dataset[train_split_name].features.get("label")

    if labels is None:
        return None, None

    id2label = (
        {idx: name.upper() for idx, name in enumerate(labels.names)}
        if hasattr(labels, "names")
        else None
    )
    label2id = (
        {name.upper(): idx for idx, name in enumerate(labels.names)}
        if hasattr(labels, "names")
        else None
    )

    return id2label, label2id


def add_class_labels(
    dataset: DatasetDict, label_field: str, label_names: list
) -> DatasetDict:
    """Add ClassLabel to dataset.

    Args:
        dataset: HuggingFace dataset
        label_field: Name of label column
        label_names: List of label names

    Returns:
        Dataset with ClassLabel column
    """
    for split in dataset.keys():
        dataset[split] = dataset[split].cast_column(
            label_field, ClassLabel(names=label_names)
        )
    return dataset
