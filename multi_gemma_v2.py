"""Generate multi-page QA pairs with Gemma 3 using a decoupled four-stage pipeline on A100.

Pipeline (no LLM-as-a-judge or automated VLM evaluation passes; designed for human annotation):
  Stage 1 — Per-page extraction (1 image at a time, cached to disk):
    Feed each page image once to Gemma 3 with SigLIP 896x896 + Pan-and-Scan high-resolution
    crops to extract structured multimodal facts:
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
    Prompt Gemma 3 in text-only mode to phrase the Indonesian question and answer
    around the two paired facts and precomputed reference answer, falling back to a
    deterministic template if phrasing validation fails after retries.

Model & Hardware Tuning (Gemma 3 on NVIDIA A100):
  - Native bfloat16 Tensor Core execution on Ampere+ (SM80+): Gemma 3 overflows in float16,
    which forced float32 on V100; on A100, bfloat16 preserves full dynamic range at 2x speed
    and half the KV-cache memory (auto-falls back to float32 on pre-Ampere GPUs).
  - Native FlashAttention-2 with CUDA graphs (`enforce_eager=False`) and prefix caching on A100
    for Gemma 3's interleaved 5:1 local sliding-window (1024) + global attention and logit softcapping.
  - SigLIP 896x896 vision encoder with Pan-and-Scan enabled by default (`--pan-and-scan`,
    `--max-crops 4`) so dense Indonesian financial tables and charts are cropped at sharp resolution
    while keeping per-page vision tokens bounded (`(1 + 4) * 256 = 1280` tokens).
  - Guided decoding off by default (`--no-guided-decoding`) with automatic `{` prefill and
    fence-safe JSON parsing to avoid `outlines` CPU FSM overhead, plus `repetition_penalty=1.05`
    to prevent Gemma 3 from looping on multi-row table JSON arrays.

Usage:
    python multi_gemma_v2.py --folder bi_multi --groups-file bi_multi_groups.txt --out bi_multipage_gemma3_4b_v4.jsonl
    python multi_gemma_v2.py --folder ojk_multi --groups-file ojk_multi_groups.txt --out ojk_multipage_gemma3_4b_v4.jsonl
    python multi_gemma_v2.py --folder bi_multi --dry-run
    python multi_gemma_v2.py --folder bi_multi --extract-only
"""

import argparse
import os

from multi_pipeline_v2 import *  # noqa: F401,F403 — re-export shared Stages 1-4 pipeline
from multi_pipeline_v2 import _patch_pil_fromarray, generate_batch, load_image, main as run_pipeline, strip_markdown_fences

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_OUT = "bi_multipage_gemma3_4b_v4.jsonl"

# Gemma 3 vision encoder: fixed 896x896 SigLIP -> 256 tokens per image (plus 256 tokens per Pan-and-Scan crop).
SIGLIP_IMAGE_SIZE = 896
SIGLIP_BASE_PIXELS = SIGLIP_IMAGE_SIZE * SIGLIP_IMAGE_SIZE  # 802,816 px
TOKENS_PER_IMAGE = 256
DEFAULT_MAX_PIXELS = 1605632  # ~2x 896x896 budget so Pan-and-Scan receives crisp source pages on A100


def format_gemma_chat(content, prefill_brace=False, processor=None):
    """Apply Gemma 3 chat template to a list of multimodal content blocks."""
    messages = [{"role": "user", "content": content}]
    prompt = None
    if processor is not None:
        try:
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = None
    if prompt is None:
        parts = []
        for item in content:
            if item.get("type") == "image":
                parts.append("<start_of_image>")
            elif item.get("type") == "text":
                parts.append(item.get("text", "").strip())
        body = "".join(parts)
        prompt = f"<bos><start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"
    if prefill_brace:
        prompt = prompt.rstrip() + "\n{"
    return prompt


def resolve_gemma_dtype(requested):
    """Select bfloat16 on Ampere+ (A100); fall back to float32 on pre-Ampere GPUs because Gemma 3 overflows in float16."""
    import torch

    has_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if requested in (None, "auto"):
        return "bfloat16" if has_bf16 else "float32"
    if requested == "half":
        replacement = "bfloat16" if has_bf16 else "float32"
        print(f"WARNING: Gemma 3 overflows in float16 ('half'); using '{replacement}' instead.")
        return replacement
    if requested == "bfloat16" and not has_bf16:
        print("WARNING: GPU does not support bfloat16; falling back to float32 for Gemma 3.")
        return "float32"
    return requested


def configure_gemma_vllm_env(attention_backend=None):
    """Use native FlashAttention-2 on A100 (SM80+); fall back to V1 FLEX_ATTENTION on pre-Ampere (V100)."""
    import torch

    has_ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    elif not has_ampere:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION")
    if not has_ampere:
        os.environ["VLLM_USE_V1"] = "1"
        if os.environ.get("VLLM_ATTENTION_BACKEND") == "FLEX_ATTENTION":
            _apply_v100_flex_attention_patch()
    return has_ampere


def _apply_v100_flex_attention_patch():
    """Patch PyTorch/vLLM FlexAttention on V100 (SM70) so V1 engine runs without Inductor CUDA-graph crashes."""
    import tempfile

    os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
    os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")
    patch_dir = tempfile.mkdtemp(prefix="vllm_flex_patch_")
    with open(os.path.join(patch_dir, "sitecustomize.py"), "w", encoding="utf-8") as f:
        f.write(
            "try:\n"
            "    import torch\n"
            "    import torch._dynamo.config as _dc\n"
            "    _dc.cache_size_limit = 512\n"
            "    _dc.accumulated_cache_size_limit = 4096\n"
            "    if hasattr(_dc, 'recompile_limit'):\n"
            "        _dc.recompile_limit = 512\n"
            "    import torch._inductor.config as _ic\n"
            "    _ic.triton.cudagraphs = False\n"
            "    _ic.triton.cudagraph_trees = False\n"
            "    import torch.nn.attention.flex_attention as _tfa\n"
            "    _orig_cbm = _tfa.create_block_mask\n"
            "    def _safe_cbm(*a, **kw):\n"
            "        torch.compiler.cudagraph_mark_step_begin()\n"
            "        kw['_compile'] = False\n"
            "        return _orig_cbm(*a, **kw)\n"
            "    _tfa.create_block_mask = _safe_cbm\n"
            "    import vllm.v1.attention.backends.flex_attention as _vfa\n"
            "    _vfa.create_block_mask = _safe_cbm\n"
            "    if hasattr(_vfa, 'create_block_mask_compiled'):\n"
            "        _vfa.create_block_mask_compiled = _safe_cbm\n"
            "    if hasattr(_vfa, 'flex_attention_compiled'):\n"
            "        _orig_fac = _vfa.flex_attention_compiled\n"
            "        def _safe_fac(*a, **kw):\n"
            "            torch.compiler.cudagraph_mark_step_begin()\n"
            "            try:\n"
            "                out = _orig_fac(*a, **kw)\n"
            "            except Exception:\n"
            "                out = _tfa.flex_attention(*a, **kw)\n"
            "            if isinstance(out, tuple):\n"
            "                return tuple(x.clone() if hasattr(x, 'clone') else x for x in out)\n"
            "            return out.clone() if hasattr(out, 'clone') else out\n"
            "        _vfa.flex_attention_compiled = _safe_fac\n"
            "except Exception:\n"
            "    pass\n"
        )
    prev_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{patch_dir}:{prev_pp}" if prev_pp else patch_dir
    try:
        import torch
        import torch._dynamo.config as _dc
        import torch._inductor.config as _ic
        import torch.nn.attention.flex_attention as _tfa
        import vllm.v1.attention.backends.flex_attention as vllm_flex

        _dc.cache_size_limit = 512
        _dc.accumulated_cache_size_limit = 4096
        if hasattr(_dc, "recompile_limit"):
            _dc.recompile_limit = 512
        _ic.triton.cudagraphs = False
        _ic.triton.cudagraph_trees = False
        _orig_cbm = _tfa.create_block_mask

        def _eager_create_block_mask(*cbm_args, **cbm_kwargs):
            torch.compiler.cudagraph_mark_step_begin()
            cbm_kwargs["_compile"] = False
            return _orig_cbm(*cbm_args, **cbm_kwargs)

        _tfa.create_block_mask = _eager_create_block_mask
        vllm_flex.create_block_mask = _eager_create_block_mask
        if hasattr(vllm_flex, "create_block_mask_compiled"):
            vllm_flex.create_block_mask_compiled = _eager_create_block_mask
    except Exception:
        pass


class HFOutput:
    def __init__(self, text, finish_reason):
        self.outputs = [type("Sample", (), {"text": text, "finish_reason": finish_reason})()]


class HFRunner:
    """Transformers fallback when vLLM is unavailable or requested via --backend hf."""

    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "half": torch.float16}
        dtype = dtype_map[resolve_gemma_dtype(args.dtype)]
        has_ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        attn_impl = "flash_attention_2" if (has_ampere and dtype == torch.bfloat16) else "sdpa"
        print(f"Loading Gemma 3 with transformers ({attn_impl}): {args.model} (dtype={dtype})")
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation=attn_impl,
                trust_remote_code=True,
            ).eval()
        except Exception:
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).eval()
        self.pan_and_scan = getattr(args, "pan_and_scan", False)
        self.max_crops = getattr(args, "max_crops", 4)

    def generate(self, inputs, params, use_tqdm=False):
        import torch

        results = []
        for item, param in zip(inputs, params):
            images = (item.get("multi_modal_data") or {}).get("image")
            proc_kwargs = {"text": item["prompt"], "return_tensors": "pt"}
            if images:
                proc_kwargs["images"] = images
                if self.pan_and_scan:
                    proc_kwargs["do_pan_and_scan"] = True
                    proc_kwargs["pan_and_scan_max_num_crops"] = self.max_crops
                    proc_kwargs["pan_and_scan_min_crop_size"] = 256
                    proc_kwargs["pan_and_scan_min_ratio_to_activate"] = 1.2
            try:
                model_inputs = self.processor(**proc_kwargs).to(self.model.device)
            except TypeError:
                for k in ("do_pan_and_scan", "pan_and_scan_max_num_crops", "pan_and_scan_min_crop_size", "pan_and_scan_min_ratio_to_activate"):
                    proc_kwargs.pop(k, None)
                model_inputs = self.processor(**proc_kwargs).to(self.model.device)

            for key, value in list(model_inputs.items()):
                if torch.is_floating_point(value):
                    model_inputs[key] = value.to(self.model.dtype)
            gen_kwargs = {
                "max_new_tokens": param.max_tokens,
                "do_sample": param.temperature > 0,
                "eos_token_id": param.stop_token_ids or None,
                "repetition_penalty": getattr(param, "repetition_penalty", 1.05),
            }
            if param.temperature > 0:
                gen_kwargs["temperature"] = param.temperature
                gen_kwargs["top_p"] = param.top_p
            with torch.inference_mode():
                output_ids = self.model.generate(**model_inputs, **gen_kwargs)
            new_tokens = output_ids[0, model_inputs["input_ids"].shape[1] :]
            text = self.processor.decode(new_tokens, skip_special_tokens=True)
            reason = "length" if len(new_tokens) >= param.max_tokens else "stop"
            results.append(HFOutput(text, reason))
        return results


class GemmaBackend:
    """Gemma 3 on vLLM, tuned for NVIDIA A100 (bfloat16, FlashAttention-2, SigLIP 896x896 + Pan-and-Scan)."""

    name = "gemma"
    description = __doc__.splitlines()[0]
    vision_knob = "--max-crops"
    image_align_factor = None
    defaults = {
        "model": DEFAULT_MODEL,
        "out": DEFAULT_OUT,
        "batch_size": 8,
        "extract_chunk": 16,
        "extract_max_pixels": DEFAULT_MAX_PIXELS,
        "max_edge": 1680,
    }

    def add_args(self, parser):
        parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
        parser.add_argument("--max-model-len", type=int, default=16384)
        parser.add_argument("--max-num-seqs", type=int, default=None,
                            help="Max concurrent sequences in vLLM (default: max(extract_chunk, batch_size * 3))")
        parser.add_argument("--gpu-util", type=float, default=None)
        parser.add_argument("--max-num-batched-tokens", type=int, default=None)
        parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float32", "half"),
                            help="Model dtype (default: auto -> bfloat16 on Ampere+/A100, float32 on V100)")
        parser.add_argument("--attention-backend", default=None,
                            help="Override VLLM_ATTENTION_BACKEND (default: native FlashAttention-2 on A100, FLEX_ATTENTION on V100)")
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--repetition-penalty", type=float, default=1.05,
                            help="Repetition penalty to prevent Gemma 3 JSON loops on dense tables (default: 1.05)")
        parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                            help="Disable CUDA graphs in vLLM (default: False on A100)")
        parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True,
                            help="Enable vLLM automatic prefix caching for shared extraction/writing prompts (default: True)")
        parser.add_argument("--pan-and-scan", action=argparse.BooleanOptionalAction, default=None,
                            help="Enable Gemma 3 SigLIP Pan-and-Scan cropping (default: auto -> True on Ampere+/A100, False on V100 V1 engine)")
        parser.add_argument("--max-crops", type=int, default=4,
                            help="Max Pan-and-Scan crops per page image (default: 4 -> at most 5x256=1280 vision tokens)")
        parser.add_argument("--guided-decoding", action=argparse.BooleanOptionalAction, default=False,
                            help="Use vLLM guided JSON decoding (default: False to avoid outlines CPU FSM overhead)")
        parser.add_argument("--guided-backend", default=None, choices=("outlines", "xgrammar", "lm-format-enforcer"))
        parser.add_argument("--prefill-brace", action=argparse.BooleanOptionalAction, default=None,
                            help="Prefill assistant turn with '{' (default: enabled when guided decoding is off)")

    def image_tokens(self, args):
        crops = args.max_crops if getattr(args, "pan_and_scan", False) else 0
        return (1 + crops) * TOKENS_PER_IMAGE

    def load_page_image(self, path, args):
        if getattr(args, "pan_and_scan", False):
            return load_image(path, args.max_edge, args.extract_max_pixels)
        # Without Pan-and-Scan, Gemma 3's SigLIP resizes the whole page to 896x896 anyway.
        return load_image(path, min(args.max_edge, SIGLIP_IMAGE_SIZE), min(args.extract_max_pixels, SIGLIP_BASE_PIXELS))

    def load(self, args):
        _patch_pil_fromarray()
        from transformers import AutoProcessor, AutoTokenizer

        self.args = args
        args.dtype = resolve_gemma_dtype(args.dtype)
        has_ampere = configure_gemma_vllm_env(args.attention_backend)
        use_flex = os.environ.get("VLLM_ATTENTION_BACKEND") == "FLEX_ATTENTION"
        if not has_ampere and not args.enforce_eager:
            args.enforce_eager = True
        if args.pan_and_scan is None:
            # vLLM V1 engine warns that do_pan_and_scan=True has suboptimal attention patterns on V1,
            # and on V100 float32 disabling pan-and-scan is 5x faster (256 vs 1280 vision tokens/image).
            args.pan_and_scan = bool(has_ampere and not use_flex)
        if args.gpu_util is None:
            args.gpu_util = 0.70 if use_flex else 0.90
        max_batched = args.max_num_batched_tokens or (2048 if use_flex else None)
        max_num_seqs = args.max_num_seqs or max(args.extract_chunk, args.batch_size * 3)

        if args.backend == "hf":
            self.llm = HFRunner(args)
        else:
            from vllm import LLM

            print(
                f"Loading Gemma 3: {args.model} | dtype={args.dtype} | "
                f"max_model_len={args.max_model_len} | max_num_seqs={max_num_seqs} | "
                f"gpu_util={args.gpu_util} | pan_and_scan={args.pan_and_scan} (max_crops={args.max_crops})"
            )
            llm_kwargs = {
                "model": args.model,
                "dtype": args.dtype,
                "max_model_len": args.max_model_len,
                "max_num_seqs": max_num_seqs,
                "gpu_memory_utilization": args.gpu_util,
                "limit_mm_per_prompt": {"image": 1},
                "enable_prefix_caching": args.prefix_caching,
                "enforce_eager": args.enforce_eager,
                "tensor_parallel_size": args.tensor_parallel_size,
                "trust_remote_code": True,
            }
            if args.pan_and_scan:
                llm_kwargs["mm_processor_kwargs"] = {
                    "do_pan_and_scan": True,
                    "pan_and_scan_max_num_crops": args.max_crops,
                    "pan_and_scan_min_crop_size": 256,
                    "pan_and_scan_min_ratio_to_activate": 1.2,
                }
            if max_batched:
                llm_kwargs["max_num_batched_tokens"] = max_batched
            try:
                self.llm = LLM(**llm_kwargs)
            except TypeError:
                llm_kwargs.pop("mm_processor_kwargs", None)
                llm_kwargs.pop("enable_prefix_caching", None)
                try:
                    self.llm = LLM(**llm_kwargs)
                except Exception as exc:
                    print(f"WARNING: vLLM failed to initialize ({exc}); falling back to transformers.")
                    self.llm = HFRunner(args)
            except Exception as exc:
                print(f"WARNING: vLLM failed to initialize ({exc}); falling back to transformers.")
                self.llm = HFRunner(args)

        if isinstance(self.llm, HFRunner):
            args.guided_decoding = False
        if args.prefill_brace is None:
            args.prefill_brace = not args.guided_decoding

        try:
            self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        except Exception:
            self.processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        vocab = tokenizer.get_vocab()
        self.stops = [
            tokenizer.convert_tokens_to_ids(t)
            for t in ("<end_of_turn>", "<eos>")
            if t in vocab
        ]

    def prompt(self, content):
        return format_gemma_chat(
            content,
            prefill_brace=bool(self.args.prefill_brace),
            processor=self.processor,
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
    run_pipeline(backend or GemmaBackend())


if __name__ == "__main__":
    main()
