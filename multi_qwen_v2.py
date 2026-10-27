"""Generate multi-page QA pairs with Qwen2.5-VL using a decoupled four-stage pipeline on A100.

Pipeline (no LLM-as-a-judge or automated VLM evaluation passes; designed for human annotation):
  Stage 1 — Per-page extraction (1 image at a time, cached to disk):
    Feed each page image once to Qwen2.5-VL at native 28x28-aligned dynamic resolution to extract
    structured multimodal facts:
      - "text": narrative statements (with or without numbers)
      - "table": table cells with row/column labels and exact numbers
      - "chart": chart/diagram series with printed data labels
    Saved to `<out>.extract.jsonl` so images are never re-encoded.
  Stage 2 — Deterministic Python verification & deduplication (per page):
    Clean extracted facts, recover missing units from statements, disambiguate table row
    subjects, parse Indonesian numbers/units, and verify cross-page uniqueness in Python.
  Stage 3 — Slot assignment & Python reference math (per 2-3 page group):
    Assign at most one question per slot with disjoint categories per group:
      - Slot 1 (q1): Text-Text Inference or Chart-Time-Sensitive
      - Slot 2 (q2): Table-Numerical Calculation or Chart-Numerical Calculation
      - Slot 3 (q3): Table-Comparison and Sorting or Chart-Information Extraction
    All numerical differences, rankings, and trend directions are computed in Python.
  Stage 4 — Text-only QA writing (no images):
    Prompt Qwen2.5-VL in text-only mode to phrase the Indonesian question and answer
    around the two paired facts and precomputed reference answer, falling back to a
    deterministic template if phrasing validation fails after retries.

Model & Hardware Tuning (Qwen2.5-VL on NVIDIA A100):
  - Native bfloat16 Tensor Core execution and FlashAttention-2 / vLLM V1 engine on Ampere+ (SM80+),
    with automatic fallback on older GPUs.
  - Prefix caching enabled and `max_num_seqs = max(extract_chunk, batch_size * 3)` so full
    extraction chunks and text-only writing batches run concurrently.
  - Image dimensions aligned to Qwen2.5-VL's 28px patch-merge grid (`IMAGE_FACTOR = 28`,
    784 pixels/token) and synced with `mm_processor_kwargs` to avoid double-resizing blur on tables.

Usage:
    python multi_qwen_v2.py --folder bi_multi --groups-file bi_multi_groups.txt --out bi_multipage_qwen25_vl_v4.jsonl
    python multi_qwen_v2.py --folder ojk_multi --groups-file ojk_multi_groups.txt --out ojk_multipage_qwen25_vl_v4.jsonl
    python multi_qwen_v2.py --folder bi_multi --dry-run
    python multi_qwen_v2.py --folder bi_multi --extract-only
"""

import argparse
import os

from multi_pipeline_v2 import *  # noqa: F401,F403 — re-export shared Stages 1-4 pipeline
from multi_pipeline_v2 import _patch_pil_fromarray, generate_batch, load_image, main as run_pipeline

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUT = "bi_multipage_qwen25_vl_v4.jsonl"

# Qwen2.5-VL vision encoder: 14x14 ViT patch + 2x2 spatial merger -> 28x28 = 784 pixels per token.
IMAGE_FACTOR = 28
PIXELS_PER_TOKEN = IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_MIN_PIXELS = 256 * PIXELS_PER_TOKEN      # 200,704 px (256 vision tokens)
DEFAULT_MAX_PIXELS = 1280 * PIXELS_PER_TOKEN     # 1,003,520 px (1,280 vision tokens; fast & sharp on V100/A100)


# --------------------------------------------------------------------------
# vLLM Execution & Qwen2.5-VL Backend (Tuned for NVIDIA A100)
# --------------------------------------------------------------------------


def resolve_qwen_dtype(requested):
    """Select bfloat16 on Ampere+ (A100), falling back to float16 on pre-Ampere GPUs."""
    import torch

    has_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if requested in (None, "auto"):
        return "bfloat16" if has_bf16 else "half"
    if requested == "bfloat16" and not has_bf16:
        print("WARNING: GPU does not support bfloat16; falling back to float16 ('half').")
        return "half"
    return requested


def configure_qwen_vllm_env(attention_backend=None):
    """Use native FlashAttention-2 / vLLM V1 on A100 (SM80+); fallback to XFormers only on pre-Ampere."""
    import torch

    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
        return
    has_ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if not has_ampere:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
        os.environ.setdefault("VLLM_USE_V1", "0")


class QwenBackend:
    """Qwen2.5-VL on vLLM, tuned for V100 / A100 with 28px-grid dynamic resolution."""

    name = "qwen"
    description = __doc__.splitlines()[0]
    vision_knob = "--extract-max-pixels"
    image_align_factor = IMAGE_FACTOR
    defaults = {
        "model": DEFAULT_MODEL,
        "out": DEFAULT_OUT,
        "batch_size": 8,
        "extract_chunk": 16,
        "extract_max_pixels": DEFAULT_MAX_PIXELS,
        "max_edge": 1680,
    }

    def add_args(self, parser):
        parser.add_argument("--extract-min-pixels", type=int, default=DEFAULT_MIN_PIXELS,
                            help="Minimum per-page pixel budget for Qwen2.5-VL processor (default: 200704 = 256 tokens)")
        parser.add_argument("--max-model-len", type=int, default=16384)
        parser.add_argument("--max-num-seqs", type=int, default=None,
                            help="Max concurrent sequences in vLLM (default: max(extract_chunk, batch_size * 3))")
        parser.add_argument("--gpu-util", type=float, default=0.90)
        parser.add_argument("--dtype", default="auto", choices=("half", "bfloat16", "float32", "auto"),
                            help="Model dtype (default: auto -> bfloat16 on Ampere+/A100, half on V100)")
        parser.add_argument("--attention-backend", default=None,
                            help="Override VLLM_ATTENTION_BACKEND (default: FlashAttention-2 on A100, XFormers on V100)")
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--repetition-penalty", type=float, default=1.05,
                            help="Repetition penalty to prevent JSON loops on dense tables (default: 1.05)")
        parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                            help="Disable CUDA graphs in vLLM (default: False)")
        parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True,
                            help="Enable vLLM automatic prefix caching for shared extraction/writing prompts (default: True)")
        parser.add_argument("--guided-decoding", action=argparse.BooleanOptionalAction, default=False,
                            help="Use vLLM guided JSON decoding (default: False to avoid outlines CPU FSM overhead)")
        parser.add_argument("--guided-backend", default=None, choices=("outlines", "xgrammar", "lm-format-enforcer"))

    def image_tokens(self, args):
        return args.extract_max_pixels // PIXELS_PER_TOKEN

    def load_page_image(self, path, args):
        return load_image(
            path,
            args.max_edge,
            args.extract_max_pixels,
            align_factor=IMAGE_FACTOR,
            min_pixels=args.extract_min_pixels,
        )

    def load(self, args):
        _patch_pil_fromarray()
        from transformers import AutoProcessor
        from vllm import LLM

        self.args = args
        args.dtype = resolve_qwen_dtype(args.dtype)
        configure_qwen_vllm_env(args.attention_backend)
        max_num_seqs = args.max_num_seqs or max(args.extract_chunk, args.batch_size * 3)

        print(
            f"Loading Qwen2.5-VL: {args.model} | dtype={args.dtype} | "
            f"max_model_len={args.max_model_len} | max_num_seqs={max_num_seqs} | "
            f"gpu_util={args.gpu_util} | max_pixels={args.extract_max_pixels}"
        )

        llm_kwargs = {
            "model": args.model,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": args.gpu_util,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "mm_processor_kwargs": {
                "min_pixels": args.extract_min_pixels,
                "max_pixels": args.extract_max_pixels,
            },
            "enable_prefix_caching": args.prefix_caching,
            "enforce_eager": args.enforce_eager,
            "tensor_parallel_size": args.tensor_parallel_size,
            "trust_remote_code": True,
        }
        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError:
            llm_kwargs.pop("mm_processor_kwargs", None)
            llm_kwargs.pop("enable_prefix_caching", None)
            self.llm = LLM(**llm_kwargs)

        self.processor = AutoProcessor.from_pretrained(
            args.model,
            min_pixels=args.extract_min_pixels,
            max_pixels=args.extract_max_pixels,
            trust_remote_code=True,
        )
        tokenizer = self.processor.tokenizer
        vocab = tokenizer.get_vocab()
        self.stops = [
            tokenizer.convert_tokens_to_ids(token)
            for token in ("<|im_end|>", "<|endoftext|>")
            if token in vocab
        ]

    def prompt(self, content):
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    def generate(self, requests):
        if not self.args.guided_decoding:
            requests = [(inputs, max_tokens, temperature, None) for inputs, max_tokens, temperature, _ in requests]
        return generate_batch(
            self.llm,
            self.stops,
            requests,
            guided_backend=self.args.guided_backend,
            repetition_penalty=self.args.repetition_penalty,
        )


def main():
    run_pipeline(QwenBackend())


if __name__ == "__main__":
    main()
