"""Reproducible 4-bit NF4 QLoRA training pipeline for Approach 2."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .qlora_config import QLoRAConfig


class MaskedSFTDataCollator:
    """Right-pad tokenized SFT records without rebuilding their loss labels."""

    def __init__(self, tokenizer: Any):
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id for SFT batching.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty SFT batch.")

        max_length = max(len(item["input_ids"]) for item in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for item in features:
            input_ids = list(item["input_ids"])
            attention_mask = list(item["attention_mask"])
            labels = list(item["labels"])
            if not (len(input_ids) == len(attention_mask) == len(labels)):
                raise ValueError("SFT batch item has misaligned token, attention, or label arrays.")
            if any(label not in (token_id, -100) for token_id, label in zip(input_ids, labels)):
                raise ValueError("SFT labels must be input token IDs or the ignored-loss value -100.")
            pad_count = max_length - len(input_ids)
            batch["input_ids"].append(input_ids + [self.pad_token_id] * pad_count)
            batch["attention_mask"].append(attention_mask + [0] * pad_count)
            batch["labels"].append(labels + [-100] * pad_count)

        return {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}


def train_qlora(
    config: QLoRAConfig,
    train_dataset: Any,
    validation_dataset: Any,
    tokenizer: Any,
    *,
    model: Any | None = None,
    trainer_cls: Any | None = None,
    run_dir: Path | str | None = None,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Execute reproducible 4-bit NF4 QLoRA fine-tuning with target-only masked loss.

    Args:
        config: QLoRA training hyperparameter configuration.
        train_dataset: Hugging Face Dataset (or dictionary) for training.
        validation_dataset: Hugging Face Dataset (or dictionary) for evaluation.
        tokenizer: Model tokenizer with configured padding and ChatML tokens.
        model: Pre-instantiated or mock model. If None, loads 4-bit quantized base model.
        trainer_cls: Optional custom or mock Trainer class. If None, uses TRL SFTTrainer.
        run_dir: Path to directory for persisting run logs and checkpoints.
        resume_from_checkpoint: Optional path to checkpoint to resume training from.

    Returns:
        Dictionary containing run_dir, adapter_dir, best_metric, and trained model.
    """
    if run_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        run_path = Path(config.output_dir) / timestamp
    else:
        run_path = Path(run_dir)

    run_path.mkdir(parents=True, exist_ok=True)
    adapter_path = run_path / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # 1. Verify target-only completion mask invariant
    if not hasattr(train_dataset, "__len__") or len(train_dataset) == 0:
        raise ValueError("Training dataset must contain at least one tokenized SFT record.")
    for index in range(len(train_dataset)):
        example = train_dataset[index]
        if not isinstance(example, dict) or not all(
            key in example for key in ("input_ids", "attention_mask", "completion_mask", "labels")
        ):
            raise ValueError(f"Training example {index} is missing token, mask, or label arrays.")
        lengths = [len(example[key]) for key in ("input_ids", "attention_mask", "completion_mask", "labels")]
        if len(set(lengths)) != 1:
            raise ValueError(f"Training example {index} has misaligned token, mask, or label arrays.")
        mask = example["completion_mask"]
        if not any(mask) or all(mask):
            raise ValueError(
                f"Training example {index} failed completion-only mask invariant: "
                "expected both masked prompt tokens and supervised target tokens."
            )
        for token_id, loss_mask, label in zip(example["input_ids"], mask, example["labels"]):
            expected = token_id if loss_mask == 1 else -100
            if label != expected:
                raise ValueError(f"Training example {index} labels do not match its completion mask.")

    # 2. Write initial run configuration
    run_metadata = {
        **config.model_dump(),
        "training_signature": config.get_training_signature(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_path),
        "adapter_dir": str(adapter_path),
    }
    config_file = run_path / "run_config.json"
    config_file.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    # 3. Model setup if not injected
    if model is None:
        import torch
        from transformers import AutoModelForCausalLM, set_seed
        from peft import get_peft_model, prepare_model_for_kbit_training

        set_seed(config.seed)
        compute_dtype = config.resolve_compute_dtype()
        quantization_config = config.get_quantization_config(compute_dtype=compute_dtype)

        active_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            quantization_config=quantization_config,
            dtype=compute_dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        active_model.config.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        active_model.config.use_cache = False

        active_model = prepare_model_for_kbit_training(
            active_model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        peft_config = config.get_peft_config()
        active_model = get_peft_model(active_model, peft_config)
    else:
        active_model = model

    # 4. Construct SFTTrainer and execute
    if trainer_cls is not None:
        trainer_factory = trainer_cls
    else:
        from trl import SFTTrainer

        trainer_factory = SFTTrainer

    training_args = config.get_sft_config(run_path, tokenizer=tokenizer)

    trainer = trainer_factory(
        model=active_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=MaskedSFTDataCollator(tokenizer),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 5. Persist adapter, tokenizer, and state
    trainer.save_model(str(adapter_path))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(adapter_path))

    best_checkpoint = None
    best_metric = None
    if hasattr(trainer, "state"):
        trainer.save_state()
        best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
        best_metric = getattr(trainer.state, "best_metric", None)

        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
            try:
                import pandas as pd

                pd.DataFrame(trainer.state.log_history).to_csv(run_path / "training_log.csv", index=False)
            except ImportError:
                (run_path / "training_log.json").write_text(
                    json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
                )

    run_metadata["best_checkpoint"] = best_checkpoint
    run_metadata["best_validation_loss"] = best_metric
    config_file.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    return {
        "run_dir": run_path,
        "adapter_dir": adapter_path,
        "best_checkpoint": best_checkpoint,
        "best_validation_loss": best_metric,
        "model": getattr(trainer, "model", active_model),
    }
