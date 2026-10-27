"""QLoRA and SFT training configuration models for Approach 2."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config import (
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_REVISION,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_LR_SCHEDULER_TYPE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_OPTIMIZER,
    DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_WARMUP_RATIO,
)


class QLoRAConfig(BaseModel):
    """Reproducible configuration for 4-bit NF4 QLoRA fine-tuning of Qwen2.5-3B.

    Adheres strictly to the Approach 2 final reference training parameters.
    """

    model_config = ConfigDict(extra="forbid")

    # Base Model Specification
    model_name: str = DEFAULT_LLM_MODEL
    model_revision: str = DEFAULT_LLM_REVISION

    # Random Seeds & Splits
    seed: int = DEFAULT_SEED
    validation_fraction: float = Field(default=DEFAULT_VALIDATION_FRACTION, gt=0.0, lt=1.0)

    # Sequence Length Budgets
    max_seq_length: int = Field(default=DEFAULT_MAX_SEQ_LENGTH, gt=0)
    max_input_tokens: int = Field(default=DEFAULT_MAX_INPUT_TOKENS, gt=0)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, gt=0)

    # LoRA Hyperparameters
    lora_rank: int = Field(default=DEFAULT_LORA_RANK, gt=0)
    lora_alpha: int = Field(default=DEFAULT_LORA_ALPHA, gt=0)
    lora_dropout: float = Field(default=DEFAULT_LORA_DROPOUT, ge=0.0, lt=1.0)
    lora_target_modules: str | list[str] = DEFAULT_LORA_TARGET_MODULES
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    lora_task_type: str = "CAUSAL_LM"

    # 4-bit NF4 Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training Hyperparameters
    num_train_epochs: int = Field(default=DEFAULT_NUM_EPOCHS, gt=0)
    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, gt=0.0)
    per_device_train_batch_size: int = Field(default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE, gt=0)
    per_device_eval_batch_size: int = Field(default=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE, gt=0)
    gradient_accumulation_steps: int = Field(default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, gt=0)
    gradient_checkpointing: bool = True
    optim: str = DEFAULT_OPTIMIZER
    lr_scheduler_type: str = DEFAULT_LR_SCHEDULER_TYPE
    warmup_ratio: float = Field(default=DEFAULT_WARMUP_RATIO, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=DEFAULT_MAX_GRAD_NORM, gt=0.0)

    # Output & Persistence
    output_dir: str = str(DEFAULT_RUNS_DIR)

    def resolve_compute_dtype(self, force_float32: bool = False) -> Any:
        """Resolve torch compute dtype based on CUDA device capabilities.

        Uses bfloat16 on Ampere or newer GPUs (capability >= 8) if supported,
        otherwise float16, or float32 for CPU/mock execution.
        """
        import torch

        if force_float32 or not torch.cuda.is_available():
            return torch.float32

        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if use_bf16 else torch.float16

    def get_quantization_config(self, compute_dtype: Any | None = None) -> Any:
        """Construct Hugging Face BitsAndBytesConfig for 4-bit NF4 loading."""
        dtype = compute_dtype or self.resolve_compute_dtype()
        try:
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=dtype,
            )
        except (ImportError, AttributeError):
            class _MockBitsAndBytesConfig:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

            return _MockBitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=dtype,
            )

    def get_peft_config(self) -> Any:
        """Construct PEFT LoraConfig for target module adaptation."""
        try:
            from peft import LoraConfig

            return LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias=self.lora_bias,
                task_type=self.lora_task_type,
            )
        except (ImportError, AttributeError):
            class _MockLoraConfig:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

            return _MockLoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias=self.lora_bias,
                task_type=self.lora_task_type,
            )

    def get_sft_config(
        self,
        run_dir: Path | str,
        *,
        tokenizer: Any | None = None,
        compute_dtype: Any | None = None,
    ) -> Any:
        """Construct TRL SFTConfig matching Approach 2 training parameters."""
        import torch

        run_path = Path(run_dir)
        dtype = compute_dtype or self.resolve_compute_dtype()
        use_bf16 = dtype == torch.bfloat16
        use_fp16 = dtype == torch.float16

        config_kwargs = {
            "output_dir": str(run_path / "checkpoints"),
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "max_length": self.max_seq_length,
            "completion_only_loss": True,
            "packing": False,
            "dataset_num_proc": 1,
            "eval_strategy": "steps",
            "eval_steps": 20,
            "save_strategy": "steps",
            "save_steps": 20,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 2,
            "prediction_loss_only": True,
            "logging_steps": 10,
            "report_to": "none",
            "push_to_hub": False,
            "seed": self.seed,
            "data_seed": self.seed,
            "eos_token": getattr(tokenizer, "eos_token", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
        }

        try:
            from trl import SFTConfig

            return SFTConfig(**config_kwargs)
        except (ImportError, AttributeError):
            class _MockSFTConfig:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

            return _MockSFTConfig(**config_kwargs)

    def get_training_signature(self, record_key: str = "") -> dict[str, Any]:
        """Generate metadata signature for run verification and reproducibility."""
        packages = {}
        for pkg in ["torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate"]:
            try:
                packages[pkg] = importlib.metadata.version(pkg)
            except Exception:
                packages[pkg] = "not-installed"

        return {
            "record_key": record_key,
            "model": self.model_name,
            "model_revision": self.model_revision,
            "seed": self.seed,
            "epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "packages": packages,
        }
