"""Generate multi-page QA pairs with InternVL2.5 using a decoupled four-stage pipeline on A100.

Pipeline (no LLM-as-a-judge or automated VLM evaluation passes; designed for human annotation):
  Stage 1 — Per-page extraction (1 image at a time, cached to disk):
    Feed each page image once to InternVL2.5 with dynamic 448x448 tiling (`InternViT-300M-448px-V2_5`)
    to extract structured multimodal facts:
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
    Table/chart slots (2, 3) are allocated before narrative slot (1) so scarce numeric facts
    are preserved for calculation/comparison questions.
  Stage 4 — Text-only QA writing (no images):
    Prompt InternVL2.5 in text-only mode to phrase the Indonesian question and answer
    around the two paired facts and precomputed reference answer, falling back to a
    deterministic template if phrasing validation fails after retries.

Model & Hardware Tuning (InternVL2.5 on NVIDIA A100):
  - Native bfloat16 Tensor Core execution and FlashAttention-2 / vLLM V1 engine on Ampere+ (SM80+),
    with automatic fallback to float16 + XFormers only on pre-Ampere GPUs (e.g., V100).
  - InternViT-300M-448px-V2_5 dynamic resolution tiling (`--max-dynamic-patch 12`): partitions
    each page into up to 12 high-res 448x448 tiles + 1 global thumbnail tile, with 0.5x pixel
    unshuffle yielding 256 visual tokens per tile (up to 3,328 tokens/page).
  - Pre-tiling pixel budget (`DEFAULT_MAX_PIXELS = 2,408,448` = 12 * 448 * 448, `max_edge = 1792`)
    aligned to 448px tile multiples so dense tables and charts are not blurred before tiling.
  - Guided decoding off by default (`--no-guided-decoding`) with automatic `{` prefill on the
    ChatML `<|im_start|>assistant` turn and fence-safe JSON parsing, plus `repetition_penalty=1.05`
    to prevent JSON loops on multi-row tables.

Usage:
    python multi_internvl_v2.py --folder bi_multi --groups-file bi_multi_groups.txt --out bi_multipage_internvl25_4b_v4.jsonl
    python multi_internvl_v2.py --folder ojk_multi --groups-file ojk_multi_groups.txt --out ojk_multipage_internvl25_4b_v4.jsonl
    python multi_internvl_v2.py --folder bi_multi --dry-run
    python multi_internvl_v2.py --folder bi_multi --extract-only
"""

import argparse
import os

from multi_pipeline_v2 import *  # noqa: F401,F403 — re-export shared Stages 1-4 pipeline
from multi_pipeline_v2 import _patch_pil_fromarray, generate_batch, load_image, main as run_pipeline, strip_markdown_fences

DEFAULT_MODEL = "OpenGVLab/InternVL2_5-4B"
DEFAULT_OUT = "bi_multipage_internvl25_4b_v4.jsonl"

# InternVL2.5 vision encoder: InternViT-300M-448px-V2_5 with 0.5x pixel-unshuffle ->
# each 448x448 tile (plus 1 thumbnail tile) produces (448 / 14 * 0.5)^2 = 256 visual tokens.
TILE_SIZE = 448
TOKENS_PER_TILE = 256
DEFAULT_MAX_DYNAMIC_PATCH = 12
DEFAULT_MAX_PIXELS = DEFAULT_MAX_DYNAMIC_PATCH * TILE_SIZE * TILE_SIZE  # 2,408,448 px (12 tiles of 448x448)


def format_internvl_chat(tokenizer, user_text, prefill_brace=False):
    """Apply InternVL2.5 ChatML template around a user message containing <image> placeholders."""
    messages = [{"role": "user", "content": user_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = (
            f"<|im_start|>system\nKamu adalah asisten yang membantu.<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    if prefill_brace:
        prompt = prompt.rstrip() + "\n{"
    return prompt


def resolve_internvl_dtype(requested):
    """Select bfloat16 on Ampere+ (A100), falling back to float16 (half) on pre-Ampere GPUs."""
    import torch

    has_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if requested in (None, "auto"):
        return "bfloat16" if has_bf16 else "half"
    if requested == "bfloat16" and not has_bf16:
        print("WARNING: GPU does not support bfloat16; falling back to float16 (half).")
        return "half"
    return requested


def configure_internvl_vllm_env(attention_backend=None):
    """Use native FlashAttention-2 / vLLM V1 on A100 (SM80+); fallback to XFormers only on pre-Ampere."""
    import torch

    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
        return
    has_ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if not has_ampere:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
        os.environ.setdefault("VLLM_USE_V1", "0")


class InternVLBackend:
    """InternVL2.5 on vLLM, tuned for NVIDIA A100 (bfloat16, FlashAttention-2, dynamic 448x448 tiling)."""

    name = "internvl"
    description = __doc__.splitlines()[0]
    vision_knob = "--max-dynamic-patch"
    image_align_factor = None
    # Dynamic tiling resizes every page onto a discrete 448px tile grid; vary temperature if multi-pass is used.
    extract_variation = "temperature"
    defaults = {
        "model": DEFAULT_MODEL,
        "out": DEFAULT_OUT,
        "batch_size": 8,
        "extract_chunk": 16,
        "extract_max_pixels": DEFAULT_MAX_PIXELS,
        "max_edge": 1792,
    }

    def add_args(self, parser):
        parser.add_argument("--max-dynamic-patch", type=int, default=DEFAULT_MAX_DYNAMIC_PATCH,
                            help="Max 448x448 tiles per page for InternViT-300M-448px-V2_5 (each tile + thumbnail = 256 tokens; default: 12)")
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
        parser.add_argument("--prefill-brace", "--prefill-json", dest="prefill_brace",
                            action=argparse.BooleanOptionalAction, default=None,
                            help="Prefill assistant turn with '{' (default: enabled when guided decoding is off)")

    def image_tokens(self, args):
        return ((args.max_dynamic_patch or DEFAULT_MAX_DYNAMIC_PATCH) + 1) * TOKENS_PER_TILE

    def load_page_image(self, path, args):
        return load_image(path, args.max_edge, args.extract_max_pixels)

    def load(self, args):
        _patch_pil_fromarray()
        from transformers import AutoTokenizer
        from vllm import LLM

        self.args = args
        args.dtype = resolve_internvl_dtype(args.dtype)
        configure_internvl_vllm_env(args.attention_backend)
        if args.prefill_brace is None:
            args.prefill_brace = not args.guided_decoding
        max_num_seqs = args.max_num_seqs or max(args.extract_chunk, args.batch_size * 3)

        print(
            f"Loading InternVL2.5: {args.model} | dtype={args.dtype} | "
            f"max_model_len={args.max_model_len} | max_num_seqs={max_num_seqs} | "
            f"gpu_util={args.gpu_util} | max_dynamic_patch={args.max_dynamic_patch} "
            f"(~{self.image_tokens(args)} tokens/page)"
        )

        llm_kwargs = {
            "model": args.model,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": args.gpu_util,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "enable_prefix_caching": args.prefix_caching,
            "enforce_eager": args.enforce_eager,
            "tensor_parallel_size": args.tensor_parallel_size,
            "trust_remote_code": True,
        }
        if args.max_dynamic_patch is not None:
            llm_kwargs["mm_processor_kwargs"] = {
                "max_dynamic_patch": args.max_dynamic_patch,
            }

        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError:
            llm_kwargs.pop("mm_processor_kwargs", None)
            llm_kwargs.pop("enable_prefix_caching", None)
            self.llm = LLM(**llm_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        vocab = self.tokenizer.get_vocab()
        self.stops = []
        for token in ("<|im_end|>", "<|endoftext|>", "</s>"):
            if token in vocab:
                tid = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(tid, int) and tid not in self.stops:
                    self.stops.append(tid)
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in self.stops:
            self.stops.append(self.tokenizer.eos_token_id)

    def prompt(self, content):
        parts = ["<image>" if block.get("type") == "image" else block["text"] for block in content]
        return format_internvl_chat(
            self.tokenizer,
            "\n".join(parts),
            prefill_brace=bool(self.args.prefill_brace),
        )

    def generate(self, requests):
        if not self.args.guided_decoding:
            requests = [(inputs, max_tokens, temperature, None) for inputs, max_tokens, temperature, _ in requests]
        outputs = generate_batch(
            self.llm,
            self.stops,
            requests,
            guided_backend=self.args.guided_backend,
            repetition_penalty=self.args.repetition_penalty,
        )
        if self.args.prefill_brace:
            normalized = []
            for text, reason in outputs:
                cleaned = strip_markdown_fences(text)
                if cleaned and not cleaned.startswith("{"):
                    cleaned = "{" + cleaned
                normalized.append((cleaned, reason))
            return normalized
        return outputs


def main(backend=None):
    run_pipeline(backend or InternVLBackend())


if __name__ == "__main__":
    main()
