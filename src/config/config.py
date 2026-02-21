"""Configuration management for demographic inference models."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: str = field(default="data")
    platform: str = field(default="twitter")
    feature: str = field(default="age")
    version: int = field(default=1)
    number_of_atleast_posts: int = field(default=20)
    number_of_posts_per_sample: int = field(default=20)
    number_of_samples: int = field(default=8)

    @property
    def train_file(self) -> str:
        """Get train file path."""
        filename = (
            f"{self.platform}_{self.feature}_train_"
            f"{self.number_of_atleast_posts}_"
            f"{self.number_of_posts_per_sample}_"
            f"{self.number_of_samples}.parquet"
        )
        return os.path.join(self.data_dir, filename)

    @property
    def test_file(self) -> str:
        """Get test file path."""
        filename = (
            f"{self.platform}_{self.feature}_test_"
            f"{self.number_of_atleast_posts}_"
            f"{self.number_of_posts_per_sample}_"
            f"{self.number_of_samples}.parquet"
        )
        return os.path.join(self.data_dir, filename)


@dataclass
class ModelConfig:
    """Model configuration."""

    model_id: str = field(default="answerdotai/ModernBERT-large")
    model_name: Optional[str] = field(default=None)
    max_seq_length: int = field(default=8000)
    num_labels: int = field(default=4)

    def get_model_name(self, data_config: DataConfig) -> str:
        """Generate model name from data config."""
        if self.model_name:
            return self.model_name
        base_name = self.model_id.split("/")[-1]
        return (
            f"{base_name}-{data_config.platform}-{data_config.feature}-"
            f"{data_config.number_of_atleast_posts}-"
            f"{data_config.number_of_posts_per_sample}-"
            f"{data_config.number_of_samples}-v{data_config.version}"
        )


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str = field(default="models")
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=5)
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=50)
    save_steps: int = field(default=50)
    eval_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)
    bf16: bool = field(default=True)
    seed: int = field(default=3407)
    use_cuda: bool = field(default=True)
    cuda_device: Optional[int] = field(default=None)

    def get_cuda_devices(self) -> Optional[str]:
        """Get CUDA device string."""
        if not self.use_cuda:
            return None
        if self.cuda_device is not None:
            return str(self.cuda_device)
        return None


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = field(default=True)
    project: str = field(default="demographic-inference")
    entity: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            wandb=WandbConfig(**config_dict.get("wandb", {})),
        )
