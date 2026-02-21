"""Causal language model trainer for demographic inference with SFT.

Uses supervised fine-tuning with causal language model objective.
Supports Gemma 12B and other causal models.
"""
import logging
import os
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, f1_score

from ..config import Config


logger = logging.getLogger(__name__)


class CausalLanguageModelTrainer:
    """Trainer for causal language models using supervised fine-tuning.

    Uses LoRA fine-tuning with 8-bit quantization.
    Suitable for text generation and completion tasks.
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
                config={
                    "max_seq_length": self.config.model.max_seq_length,
                    "train_batch_size": self.config.training.per_device_train_batch_size,
                    "learning_rate": self.config.training.learning_rate,
                    "num_epochs": self.config.training.num_train_epochs,
                },
            )

    def train(self):
        """Run supervised fine-tuning."""
        # Setup quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else None,
        )

        # Setup LoRA
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_id)
        tokenizer.pad_token = tokenizer.eos_token

        # Define prompt format for SFT
        def formatting_func(examples):
            """Format examples for causal language model training."""
            texts = []
            for instruction, input_text, output in zip(
                examples.get("instruction", [""]*len(examples["input_data"])),
                examples["input_data"],
                examples["output_data"],
            ):
                input_text = input_text.replace("[removed]", "")
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
                texts.append(prompt)
            return {"text": texts}

        # Load dataset
        dataset = load_dataset(
            "parquet", data_files=self.config.data.train_file, split="train"
        )
        dataset = dataset.map(formatting_func, batched=True)

        # Setup trainer
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=self.config.training.warmup_steps,
            num_train_epochs=self.config.training.num_train_epochs,
            learning_rate=self.config.training.learning_rate,
            bf16=self.config.training.bf16,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=self.config.training.weight_decay,
            lr_scheduler_type="linear",
            save_steps=50,
            eval_steps=50,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        # Train
        trainer.train()

        # Save model
        model_name = self.config.model.get_model_name(self.config.data)
        model_path = os.path.join(self.config.training.output_dir, model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        logger.info(f"Model saved to {model_path}")

        # Log to W&B
        if self.config.wandb.enabled:
            artifact = wandb.Artifact(f"{model_name}_artifact", type="model")
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)

    def evaluate(self):
        """Run evaluation on test set."""
        # Load test dataset
        test_dataset = load_dataset(
            "parquet", data_files=self.config.data.test_file, split="train"
        )

        def format_test_prompts(examples):
            """Format test examples for inference."""
            texts = []
            for instruction, input_text in zip(
                examples.get("instruction", [""]*len(examples["input_data"])),
                examples["input_data"],
            ):
                input_text = input_text.replace("[removed]", "")
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
                texts.append(prompt)
            return {"text": texts}

        test_dataset = test_dataset.map(format_test_prompts, batched=True)
        test_df = test_dataset.data.to_pandas()

        # Load model for inference
        model_name = self.config.model.get_model_name(self.config.data)
        model_path = os.path.join(self.config.training.output_dir, model_name)

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Generate predictions
        test_df["llm_output"] = None
        for i, data_point in tqdm(test_df.iterrows(), total=len(test_df)):
            inputs = tokenizer(
                data_point["text"], return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_text = tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
            )
            test_df.loc[i, "llm_output"] = generated_text.strip()

        # Calculate metrics
        accuracy = accuracy_score(
            test_df["output_data"].str.strip(), test_df["llm_output"].str.strip()
        )
        f1 = f1_score(
            test_df["output_data"].str.strip(),
            test_df["llm_output"].str.strip(),
            average="macro",
            zero_division=0,
        )
        f1_weighted = f1_score(
            test_df["output_data"].str.strip(),
            test_df["llm_output"].str.strip(),
            average="weighted",
            zero_division=0,
        )

        logger.info(
            f"Accuracy: {accuracy:.4f}, F1 (macro): {f1:.4f}, "
            f"F1 (weighted): {f1_weighted:.4f}"
        )

        if self.config.wandb.enabled:
            wandb.log({
                "accuracy": accuracy,
                "f1_score": f1,
                "f1_weighted": f1_weighted,
            })
            wandb.finish()
