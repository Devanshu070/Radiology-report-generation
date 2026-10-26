"""Target-only completion masking and ChatML token alignment for Qwen SFT."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any, Literal

from ..config import DEFAULT_LLM_MODEL, DEFAULT_LLM_REVISION
from ..schemas import FormattedSFTExample, TokenizedSFTRecord
from .validation import validate_tokenized_sft_record

IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643


class MockQwenTokenizer:
    """Offline mock tokenizer implementing Qwen ChatML formatting and tokenization."""

    def __init__(self):
        self.pad_token_id = PAD_TOKEN_ID
        self.eos_token_id = IM_END_TOKEN_ID
        self.im_start_token_id = IM_START_TOKEN_ID
        self.im_end_token_id = IM_END_TOKEN_ID
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|im_end|>"
        self.padding_side = "right"
        self.vocab = {
            "<|im_start|>": IM_START_TOKEN_ID,
            "<|im_end|>": IM_END_TOKEN_ID,
            "<|endoftext|>": PAD_TOKEN_ID,
            "system": 1001,
            "user": 1002,
            "assistant": 1003,
            "\n": 1004,
        }
        self._reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._next_id = 2000

    def _get_token_id(self, word: str) -> int:
        if word in self.vocab:
            return self.vocab[word]
        token_id = self._next_id
        self.vocab[word] = token_id
        self._reverse_vocab[token_id] = word
        self._next_id += 1
        return token_id

    def tokenize_text(self, text: str) -> list[int]:
        """Simple deterministic word-level tokenization for offline testing."""
        tokens: list[int] = []
        # Pattern matching special tokens or whitespace-separated words/punctuation
        parts = re.findall(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\n|\S+", text)
        for part in parts:
            tokens.append(self._get_token_id(part))
        return tokens

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int] | str:
        """Format messages following the Qwen ChatML standard."""
        rendered = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]
            rendered += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"

        if tokenize:
            return self.tokenize_text(rendered)
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenize_text(text)

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        words = []
        for tid in token_ids:
            w = self._reverse_vocab.get(tid, f"<token_{tid}>")
            if skip_special_tokens and w.startswith("<|"):
                continue
            words.append(w)
        return " ".join(words)

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        batch_ids = [self.tokenize_text(t) for t in texts]

        if truncation and max_length is not None:
            batch_ids = [ids[:max_length] for ids in batch_ids]

        max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0
        if padding == "max_length" and max_length is not None:
            max_len = max_length
        elif not padding:
            max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0

        padded_ids = []
        attention_masks = []
        for ids in batch_ids:
            cur_len = len(ids)
            pad_len = max(0, max_len - cur_len)
            padded = ids + [self.pad_token_id] * pad_len
            mask = [1] * cur_len + [0] * pad_len
            padded_ids.append(padded)
            attention_masks.append(mask)

        return {
            "input_ids": padded_ids[0] if isinstance(text, str) else padded_ids,
            "attention_mask": attention_masks[0] if isinstance(text, str) else attention_masks,
        }


def get_qwen_tokenizer(
    model_name: str = DEFAULT_LLM_MODEL,
    revision: str = DEFAULT_LLM_REVISION,
    *,
    tokenizer: Any | None = None,
) -> Any:
    """Load or inject the Qwen tokenizer.

    If an explicit tokenizer is provided, it is returned directly;
    otherwise Hugging Face AutoTokenizer is loaded with right-padding.
    """
    if tokenizer is not None:
        return tokenizer

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def tokenize_sft_example(
    example: FormattedSFTExample,
    tokenizer: Any,
    *,
    max_length: int | None = None,
    padding: bool | Literal["max_length", "longest"] = False,
    pad_to_multiple_of: int | None = None,
) -> TokenizedSFTRecord:
    """Tokenize an SFT example and produce target-only completion and label masks.

    Labels are set to ``-100`` for all prompt tokens (system instructions,
    retrieved demonstrations, current case context, and the ``<|im_start|>assistant\n``
    header) as well as padding tokens. Only assistant target completion tokens
    receive active loss labels.

    Args:
        example: FormattedSFTExample with prompt_messages and optional completion_message.
        tokenizer: Hugging Face or Mock tokenizer supporting apply_chat_template.
        max_length: Maximum allowed sequence length.
        padding: Padding strategy (False, True, 'max_length', 'longest').
        pad_to_multiple_of: Optional integer alignment for padding length.

    Returns:
        Validated TokenizedSFTRecord with equal-length token and mask arrays.
    """
    prompt_dicts = [m.model_dump() for m in example.prompt_messages]

    prompt_token_ids = tokenizer.apply_chat_template(
        prompt_dicts,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_len = len(prompt_token_ids)

    if example.completion_message is not None:
        all_dicts = [*prompt_dicts, example.completion_message.model_dump()]
        full_token_ids = tokenizer.apply_chat_template(
            all_dicts,
            tokenize=True,
            add_generation_prompt=False,
        )
    else:
        full_token_ids = list(prompt_token_ids)

    full_len = len(full_token_ids)

    # Determine completion boundary
    # In ChatML, prompt with add_generation_prompt=True covers up to <|im_start|>assistant\n
    # The completion turn starts immediately after prompt_len.
    boundary_idx = min(prompt_len, full_len)

    completion_mask: list[int] = [0] * boundary_idx + [1] * (full_len - boundary_idx)
    labels: list[int] = [-100] * boundary_idx + full_token_ids[boundary_idx:]
    attention_mask: list[int] = [1] * full_len
    input_ids: list[int] = list(full_token_ids)

    # Truncation if sequence exceeds max_length
    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        completion_mask = completion_mask[:max_length]
        labels = labels[:max_length]

    if example.completion_message is not None and not any(completion_mask):
        raise ValueError(
            f"Truncation removed every supervised completion token for case '{example.case_id}'."
        )

    # Padding calculation
    target_length = len(input_ids)
    if padding == "max_length" and max_length is not None:
        target_length = max_length
    elif padding:
        target_length = len(input_ids)

    if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
        remainder = target_length % pad_to_multiple_of
        if remainder != 0:
            target_length += pad_to_multiple_of - remainder

    if target_length > len(input_ids):
        pad_count = target_length - len(input_ids)
        pad_token_id = getattr(tokenizer, "pad_token_id", PAD_TOKEN_ID)
        if pad_token_id is None:
            pad_token_id = PAD_TOKEN_ID

        input_ids.extend([pad_token_id] * pad_count)
        attention_mask.extend([0] * pad_count)
        completion_mask.extend([0] * pad_count)
        labels.extend([-100] * pad_count)

    record = TokenizedSFTRecord(
        case_id=example.case_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        labels=labels,
    )

    return validate_tokenized_sft_record(record)


def tokenize_sft_dataset(
    examples: Sequence[FormattedSFTExample],
    tokenizer: Any,
    *,
    max_length: int | None = None,
    padding: bool | Literal["max_length", "longest"] = False,
    pad_to_multiple_of: int | None = None,
) -> list[TokenizedSFTRecord]:
    """Tokenize and mask SFT examples, padding longest batches consistently."""
    records = [
        tokenize_sft_example(
            ex,
            tokenizer,
            max_length=max_length,
            padding=False if padding in (True, "longest") else padding,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        for ex in examples
    ]

    if padding in (True, "longest") and records:
        target_length = max(len(record.input_ids) for record in records)
        if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
            target_length = (
                (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of
        pad_token_id = getattr(tokenizer, "pad_token_id", PAD_TOKEN_ID)
        if pad_token_id is None:
            pad_token_id = PAD_TOKEN_ID
        for record in records:
            pad_count = target_length - len(record.input_ids)
            if pad_count:
                record.input_ids.extend([pad_token_id] * pad_count)
                record.attention_mask.extend([0] * pad_count)
                record.completion_mask.extend([0] * pad_count)
                record.labels.extend([-100] * pad_count)
                validate_tokenized_sft_record(record)
    return records
