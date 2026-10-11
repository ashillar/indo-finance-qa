"""Generate multi-page QA pairs from page images with Gemma 3 and vLLM.

Same pipeline as multi_internvl_fixed.py (page pairs per question, evidence
per page, duplicate and vagueness checks, per-question retries, shared retry
queue, --groups-file, --dry-run, --shard). Only the model-specific parts are
replaced: the prompt format, the image-token budget and how vLLM is started.

    python multi_gemma.py --folder bi_multi --groups-file bi_multi_groups.txt
    python multi_gemma.py --model google/gemma-3-12b-it --no-pan-and-scan

Gemma 3 reads every image at 896x896 (256 tokens). Dense report pages lose
detail at that size, so pan-and-scan is on by default: the page is also cut
into up to 4 crops, so one page costs at most 5 * 256 = 1280 tokens.
--no-pan-and-scan costs 256 tokens per page but reads small print worse.

The model is gated on Hugging Face: accept the license and run
`huggingface-cli login` first. V100 has no bfloat16, and Gemma 3 can give
NaN or garbage in float16; if that happens use --dtype float32 (4B only).
"""

import argparse
import sys

import multi_internvl_fixed as base

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_OUT = "bi_multipage_gemma3_v2.jsonl"
STOP_TOKENS = ("<end_of_turn>", "<eos>")

options = argparse.Namespace(pan_and_scan=True, dtype="half")


class GemmaPrompt:
    """Stands in for the tokenizer in build_group_prompt: turns its
    "<image>" placeholders and single user message into Gemma's format."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = messages[0]["content"].replace("<image>", "<start_of_image>")
        return (
            f"<bos><start_of_turn>user\n{text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )


def load_tokenizer_and_stops(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    stops = []
    for token in STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            token_id is not None
            and token_id >= 0
            and token_id != getattr(tokenizer, "unk_token_id", None)
        ):
            stops.append(token_id)
    return GemmaPrompt(), list(dict.fromkeys(stops))


def make_llm(args, group_size):
    from vllm import LLM

    return LLM(
        model=args.model,
        dtype=options.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        limit_mm_per_prompt={"image": group_size},
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=False,
        mm_processor_kwargs={"do_pan_and_scan": options.pan_and_scan},
    )


base_parse_args = base.parse_args


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pan-and-scan", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dtype", default="half", choices=("half", "float32", "auto")
    )
    known, rest = parser.parse_known_args()
    options.pan_and_scan = known.pan_and_scan
    options.dtype = known.dtype
    sys.argv = [sys.argv[0]] + rest
    # base computes tokens per image as (max_dynamic_patch + 1) * 256 when
    # the value is above 1, and 256 otherwise: 4 crops + the full page, or 1.
    sys.argv += ["--max-dynamic-patch", "4" if known.pan_and_scan else "1"]
    if "--out" not in rest:
        sys.argv += ["--out", DEFAULT_OUT]
    if "--model" not in rest:
        sys.argv += ["--model", DEFAULT_MODEL]
    return base_parse_args()


base.parse_args = parse_args
base.load_tokenizer_and_stops = load_tokenizer_and_stops
base.make_llm = make_llm


if __name__ == "__main__":
    base.main()
