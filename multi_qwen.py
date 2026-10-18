"""Generate multi-page QA pairs from document page images with Qwen2.5-VL and vLLM.

Usage:
    python multi_qwen.py --folder bi_multi --groups-file bi_multi_groups.txt --out bi_multipage_qwen25_vl.jsonl
    python multi_qwen.py --folder ojk_multi --groups-file ojk_multi_groups.txt --out ojk_multipage_qwen25_vl.jsonl
    python multi_qwen.py --folder bi_multi --dry-run
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

# V100 compatibility: XFormers attention backend and vLLM V0 engine.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_V1", "0")

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUT = "bi_multipage_qwen25_vl.jsonl"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)
# Qwen2.5-VL: 14px patch, 2x2 spatial merge -> 28x28 pixels per vision token.
PIXELS_PER_TOKEN = 28 * 28

CATEGORIES = [
    "Text-Text Inference",
    "Chart-Information Extraction",
    "Chart-Numerical Calculation",
    "Chart-Time-Sensitive",
    "Table-Numerical Calculation",
    "Table-Comparison and Sorting",
]
ANSWER_TYPES = ["short", "long"]
ITEM_KEYS = ("query", "category", "answer", "answer_type", "source_path")

# Reject explicit references to table/figure numbers in queries.
FIG_TABLE_NUM_RE = re.compile(
    r"\b(?:tabel|grafik|gambar|bagan|figure|table|chart)\s+"
    r"(?:no\.?\s*)?(?:\d+|[ivxlcdm]+(?:\.\d+)*)\b",
    re.IGNORECASE,
)

PROMPT_TEMPLATE = """Here are images of a document ({doc_name}). Each image is labeled with its exact original file path (`source_path`):
{source_list}

Your task is to generate question-answer pairs about the document from various perspectives in Indonesian (Bahasa Indonesia). Categorize each question using `category`, provide its answer using `answer`, specify whether the answer is `long` or `short` using `answer_type`, and include the original document path using `source_path`.

Instructions
- Generate exactly three different questions for each document.
- Write all `query` and `answer` texts in Bahasa Indonesia.
- Since this is a multi-page document group, prioritize questions that synthesize, compare, or connect information across multiple pages whenever the pages share related content, tables, or charts.
- Make questions detailed and specific. Begin each question by identifying the document and describing the relevant text, table, chart, or other content. Do not explicitly mention figure or table numbers.
- Questions may target text, tables, charts, graphs, or other visual elements.
- Ensure the three questions for each document have distinct categories.

Text-based QA
- Text-Text Inference: Information extraction or reasoning based on textual content.

Chart-based QA
- Chart-Information Extraction: Extract key metrics, values, or features from charts.
- Chart-Numerical Calculation: Calculate growth rates, percentages, totals, differences, or other numerical values from chart data.
- Chart-Time-Sensitive: Analyze trends, changes over time, event sequences, frequencies, or durations.

Table-based QA
- Table-Numerical Calculation: Calculate totals, percentages, growth rates, differences, or ratios using table data.
- Table-Comparison and Sorting: Compare or rank entities according to specified criteria using table data.

- Answers must be accurate and supported by the document. Never invent information. Show calculations and units when appropriate.
- Use `short` for concise answers and `long` for detailed explanations or calculations.
- Select categories appropriate to the available content.
- Set `source_path` to the exact original file path provided by the input pipeline above. Do not invent, modify, or guess paths. If the path is not provided, use an empty string.
- The `source_path` must identify the original source file used to answer the question. If a question uses multiple files, store all relevant paths as a JSON array of strings instead of a single string.
- Each object must contain exactly five keys: `query`, `category`, `answer`, `answer_type`, and `source_path`.
- Return valid JSON only, without Markdown code fences or additional explanations.
{avoid_block}{feedback_block}
Output Format

{{ "result": [ {{ "query": "Detailed question in Bahasa Indonesia identifying the document and relevant content.", "category": "Text-Text Inference", "answer": "Evidence-based answer in Bahasa Indonesia.", "answer_type": "long", "source_path": [{example_multi_paths}] }}, {{ "query": "Detailed question in Bahasa Indonesia about numerical information in a table.", "category": "Table-Numerical Calculation", "answer": "Answer in Bahasa Indonesia with the calculation when appropriate.", "answer_type": "short", "source_path": [{example_multi_paths}] }}, {{ "query": "Detailed question in Bahasa Indonesia about chart information.", "category": "Chart-Information Extraction", "answer": "Accurate answer in Bahasa Indonesia supported by the document.", "answer_type": "short", "source_path": "{example_single_path}" }} ] }}"""


# --------------------------------------------------------------------------
# Helpers: text, JSON, paths, and page grouping
# --------------------------------------------------------------------------


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_near_duplicate(query, previous, threshold):
    norm = normalize_text(query)
    return any(
        difflib.SequenceMatcher(None, norm, other).ratio() >= threshold
        for other in previous
    )


def parse_json(text):
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model output")
    try:
        return json.JSONDecoder().raw_decode(cleaned[start:])[0]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def normalize_name(name):
    """Repair filenames where percent encoding lost its '%' sign (' 20' -> ' ')."""
    if name.count(" 20") >= 2:
        name = name.replace(" 20", " ")
    return re.sub(r"\s+", " ", urllib.parse.unquote(name)).strip()


def format_source_path(path, path_mode):
    if path_mode == "resolve":
        return str(path.resolve())
    if path_mode == "name":
        return path.name
    return str(path)


def group_documents(folder):
    docs = defaultdict(dict)
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = PAGE_PATTERN.match(normalize_name(path.stem))
        if match:
            doc = match.group("doc").strip().rstrip("_").strip()
            docs[doc][int(match.group("page"))] = path
    return docs


def parse_group_spec(spec):
    """Turn '38-39', '45-47' or '38,39,41' into a sorted list of page numbers."""
    pages = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            pages.update(range(low, high + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def read_groups_file(path):
    entries = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                sys.exit(f"ERROR: {path} line {number}: expected '<doc>: 38-39'")
            doc, specs = line.rsplit(":", 1)
            entries.append((doc.strip(), specs.replace(";", " ").split()))
    return entries


def split_runs(pages, max_gap):
    runs = [[pages[0]]]
    for page in pages[1:]:
        if page - runs[-1][-1] <= max_gap:
            runs[-1].append(page)
        else:
            runs.append([page])
    return runs


def make_windows(run, size, overlap):
    if len(run) <= size:
        return [run]
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(run), step):
        window = run[start : start + size]
        if len(window) < 2:
            window = run[-2:]
        windows.append(window)
        if start + size >= len(run):
            break
    return windows


def auto_entries(docs, group_size, overlap, max_page_gap):
    entries = []
    for doc in sorted(docs):
        pages = sorted(docs[doc])
        specs = []
        for run in split_runs(pages, max_page_gap):
            if len(run) < 2:
                continue
            for window in make_windows(run, group_size, overlap):
                specs.append(",".join(map(str, window)))
        if specs:
            entries.append((doc, specs))
    return entries


def build_units(docs, entries, quiet=False):
    units, seen = [], set()
    for fragment, specs in entries:
        name = fragment.lower()
        matches = [d for d in docs if d.lower() == name] or [
            d for d in docs if name in d.lower()
        ]
        if not matches:
            if not quiet:
                print(f"WARNING: no document matches {fragment!r}")
        for doc in matches:
            for spec in specs:
                pages = parse_group_spec(spec)
                missing = [p for p in pages if p not in docs[doc]]
                if missing or len(pages) < 2:
                    if not quiet:
                        print(
                            f"WARNING: {doc} {spec} skipped "
                            f"({'missing ' + str(missing) if missing else '<2 pages'})"
                        )
                    continue
                key = f"{doc}::" + "-".join(f"p{p:03d}" for p in pages)
                if key not in seen:
                    seen.add(key)
                    units.append((key, doc, [(p, docs[doc][p]) for p in pages]))
    return units


def load_image(path, max_edge, max_pixels):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_edge / max(width, height))
    if width * height * scale**2 > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
    if scale < 1.0:
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(size, Image.LANCZOS)
    return image


# --------------------------------------------------------------------------
# Prompt & Schema Construction
# --------------------------------------------------------------------------


def build_schema(source_paths):
    path_enum = list(dict.fromkeys(source_paths))
    source_path_schema = {
        "anyOf": [
            {"type": "string", "enum": path_enum},
            {
                "type": "array",
                "items": {"type": "string", "enum": path_enum},
                "minItems": 1,
                "maxItems": len(path_enum),
            },
        ]
    }
    item_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string", "enum": CATEGORIES},
            "answer": {"type": "string"},
            "answer_type": {"type": "string", "enum": ANSWER_TYPES},
            "source_path": source_path_schema,
        },
        "required": list(ITEM_KEYS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "result": {
                "type": "array",
                "items": item_schema,
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["result"],
        "additionalProperties": False,
    }


def build_request(processor, job, avoid_queries):
    source_paths = job["source_paths"]
    source_list = "\n".join(
        f"- Page {page}: `{sp}`"
        for (page, _), sp in zip(job["window"], source_paths)
    )
    example_single = source_paths[0] if source_paths else ""
    example_multi = ", ".join(f'"{p}"' for p in source_paths[:2])

    avoid_block = ""
    if avoid_queries:
        avoid_block = (
            "\nAvoid repeating these questions already generated for this document:\n"
            + "\n".join(f"- {q}" for q in avoid_queries[-9:])
            + "\n"
        )

    feedback_block = ""
    if job["feedback"]:
        feedback_block = (
            f"\nPREVIOUS ATTEMPT REJECTED: {job['feedback']}\n"
            "Fix all reported issues and regenerate the JSON output.\n"
        )

    prompt_text = PROMPT_TEMPLATE.format(
        doc_name=job["doc"],
        source_list=source_list,
        avoid_block=avoid_block,
        feedback_block=feedback_block,
        example_single_path=example_single,
        example_multi_paths=example_multi,
    )

    content = []
    for (page, _), sp in zip(job["window"], source_paths):
        content.append({"type": "image"})
        content.append({"type": "text", "text": f"[Page {page} | source_path: {sp}]"})
    content.append({"type": "text", "text": prompt_text})

    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": [job["images"][p] for p in job["pages"]]},
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def resolve_one_source_path(raw_val, valid_paths, pages_to_path):
    """Match a model-returned path string against the group's exact source_paths."""
    if not isinstance(raw_val, str):
        raw_val = str(raw_val or "")
    candidate = raw_val.strip()
    if not candidate:
        return None
    if candidate in valid_paths:
        return candidate
    # Match by basename or normalized basename
    cand_base = os.path.basename(candidate)
    norm_cand = normalize_name(Path(cand_base).stem).lower()
    for vp in valid_paths:
        if os.path.basename(vp) == cand_base:
            return vp
        if normalize_name(Path(vp).stem).lower() == norm_cand:
            return vp
    # Match by page number if model wrote e.g. "p014" or "14"
    page_match = re.search(r"p?0*(\d+)$", cand_base, re.IGNORECASE)
    if page_match:
        page_num = int(page_match.group(1))
        if page_num in pages_to_path:
            return pages_to_path[page_num]
    return None


def normalize_source_path(raw_source, valid_paths, pages_to_path):
    """Validate and normalize source_path to a string (1 file) or list of strings (>1 files)."""
    if not valid_paths:
        return ""
    if isinstance(raw_source, str):
        raw_list = [raw_source]
    elif isinstance(raw_source, list):
        raw_list = raw_source
    else:
        return None

    resolved = []
    for entry in raw_list:
        matched = resolve_one_source_path(entry, valid_paths, pages_to_path)
        if matched is None:
            return None
        if matched not in resolved:
            resolved.append(matched)

    if not resolved:
        return None
    # Sort according to group page order
    order = {path: idx for idx, path in enumerate(valid_paths)}
    resolved.sort(key=lambda p: order.get(p, 999))
    return resolved[0] if len(resolved) == 1 else resolved


def normalize_answer_type(raw_type, answer):
    val = str(raw_type or "").strip().lower()
    if val in ("short", "pendek"):
        return "short"
    if val in ("long", "panjang"):
        return "long"
    return "long" if len(answer) > 100 else "short"


def validate_result(data, job, seen_queries, dup_threshold, require_multipage, is_final):
    if not isinstance(data, dict):
        return "top-level JSON must be an object", None, []
    result = data.get("result")
    if not isinstance(result, list) or len(result) != 3:
        return "JSON 'result' must be a list of exactly 3 question-answer objects", None, []

    valid_paths = job["source_paths"]
    pages_to_path = {
        page: sp for (page, _), sp in zip(job["window"], valid_paths)
    }

    cleaned_items = []
    used_categories = set()
    batch_queries = []
    duplicates = []
    has_multipage_item = False

    for idx, item in enumerate(result, 1):
        if not isinstance(item, dict):
            return f"item {idx} is not an object", None, []

        query = str(item.get("query") or "").strip()
        category = str(item.get("category") or "").strip()
        answer = str(item.get("answer") or "").strip()
        raw_type = item.get("answer_type")
        raw_source = item.get("source_path")

        if len(query) < 15:
            return f"item {idx} query is too short or empty", None, []
        if len(answer) < 2:
            return f"item {idx} answer is empty", None, []
        if FIG_TABLE_NUM_RE.search(query):
            return (
                f"item {idx} query explicitly mentions a figure or table number; "
                "describe the content instead"
            ), None, []
        if category not in CATEGORIES:
            return (
                f"item {idx} has invalid category {category!r}; "
                f"must be one of {CATEGORIES}"
            ), None, []
        if category in used_categories:
            return (
                f"duplicate category {category!r}; all 3 questions must have "
                "distinct categories"
            ), None, []
        used_categories.add(category)

        norm_source = normalize_source_path(raw_source, valid_paths, pages_to_path)
        if norm_source is None:
            return (
                f"item {idx} has invalid source_path {raw_source!r}; "
                f"use exact paths from {valid_paths}"
            ), None, []
        if isinstance(norm_source, list) and len(norm_source) >= 2:
            has_multipage_item = True

        if is_near_duplicate(query, batch_queries, dup_threshold):
            return f"item {idx} repeats another question in the same set", None, []
        if is_near_duplicate(query, seen_queries, dup_threshold):
            if not is_final:
                return f"item {idx} repeats a question from an earlier group", None, []
            duplicates.append(query)

        batch_queries.append(normalize_text(query))
        answer_type = normalize_answer_type(raw_type, answer)

        # Keep exact 5 keys required by the output format specification.
        cleaned_items.append(
            {
                "query": query,
                "category": category,
                "answer": answer,
                "answer_type": answer_type,
                "source_path": norm_source,
            }
        )

    if require_multipage and len(valid_paths) >= 2 and not has_multipage_item and not is_final:
        return (
            "at least one question must synthesize information across multiple pages "
            "(source_path as a JSON array of multiple paths)"
        ), None, []

    return None, cleaned_items, duplicates


# --------------------------------------------------------------------------
# vLLM Execution
# --------------------------------------------------------------------------


def load_model(args, max_pages):
    from transformers import AutoProcessor
    from vllm import LLM

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        limit_mm_per_prompt={"image": max_pages},
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    stops = [
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<|im_end|>", "<|endoftext|>")
        if token in tokenizer.get_vocab()
    ]
    return llm, processor, stops


def make_sampling_params(max_tokens, temperature, stops, schema=None):
    from vllm import SamplingParams

    common = {
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "max_tokens": max_tokens,
        "stop_token_ids": stops,
    }
    if schema is None:
        return SamplingParams(**common)
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            **common, guided_decoding=GuidedDecodingParams(json=schema)
        )
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **common, structured_outputs=StructuredOutputsParams(json=schema)
        )


def generate_batch(llm, stops, requests):
    if not requests:
        return []
    inputs = [req[0] for req in requests]
    try:
        params = [make_sampling_params(m, t, stops, s) for _, m, t, s in requests]
        outputs = llm.generate(inputs, params, use_tqdm=False)
    except Exception as exc:
        if not re.search(r"schema|grammar|guided|structured", str(exc), re.IGNORECASE):
            raise
        print(f"WARNING: guided JSON decoding unavailable ({exc}); running without schema")
        params = [make_sampling_params(m, t, stops, None) for _, m, t, _ in requests]
        outputs = llm.generate(inputs, params, use_tqdm=False)
    return [
        (o.outputs[0].text, o.outputs[0].finish_reason) if o.outputs else ("", None)
        for o in outputs
    ]


# --------------------------------------------------------------------------
# CLI & Main Loop
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", required=True, help="Folder containing page images")
    parser.add_argument(
        "--groups-file",
        default=None,
        help="Page groups file (default: <folder>_groups.txt if present, else auto-window)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--doc", help="Filter to documents whose name contains this substring")
    parser.add_argument("--limit", type=int, help="Process at most N groups")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1003520,
        help="Per-image pixel cap (~1280 vision tokens for Qwen2.5-VL)",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--text-token-reserve", type=int, default=1200)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--dtype", default="half", choices=("half", "bfloat16", "float32", "auto"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dup-threshold", type=float, default=0.85)
    parser.add_argument(
        "--path-mode",
        choices=("as-given", "resolve", "name"),
        default="as-given",
        help="How source_path is formatted: 'as-given' (relative to --folder), 'resolve' (absolute), or 'name'",
    )
    parser.add_argument(
        "--require-multipage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require at least one question per group to reference multiple pages in source_path",
    )
    parser.add_argument(
        "--raw-json-only",
        action="store_true",
        help="Write only {'result': [...]} per JSONL line without group metadata wrapper",
    )
    parser.add_argument("--group-size", type=int, default=3, help="Auto-window size when no groups-file is used")
    parser.add_argument("--overlap", type=int, default=1, help="Auto-window overlap when no groups-file is used")
    parser.add_argument("--max-page-gap", type=int, default=2, help="Max page gap when auto-grouping")
    parser.add_argument("--dry-run", action="store_true", help="List page groups and exit without loading vLLM")
    return parser.parse_args()


def load_done(path):
    done, seen = set(), defaultdict(list)
    if not path.exists():
        return done, seen
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        qa = record.get("qa") if "qa" in record else record
        if isinstance(qa, dict) and isinstance(qa.get("result"), list) and qa["result"]:
            if "group_key" in record:
                done.add(record["group_key"])
            doc = record.get("document", "")
            seen[doc].extend(
                normalize_text(item.get("query", ""))
                for item in qa["result"]
                if isinstance(item, dict)
            )
    return done, seen


def write_record(stream, job, attempt, items, duplicates, error, raw_json_only):
    if raw_json_only:
        payload = {"result": items or []}
    else:
        payload = {
            "group_key": job["key"],
            "document": job["doc"],
            "pages": job["pages"],
            "source_paths": job["source_paths"],
            "attempts": attempt,
            "qa": {"result": items} if items is not None else None,
        }
        if duplicates:
            payload["near_duplicate_queries"] = duplicates
        if error:
            payload["error"] = error
            payload["raw_output"] = job["raw"]
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def main():
    args = parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"ERROR: folder does not exist: {folder}")

    docs = group_documents(folder)
    if args.doc:
        docs = {d: p for d, p in docs.items() if args.doc.lower() in d.lower()}

    groups_file = args.groups_file
    if groups_file:
        units = build_units(docs, read_groups_file(groups_file))
    else:
        default_groups = Path(f"{folder.name}_groups.txt")
        units = []
        if default_groups.exists():
            units = build_units(
                docs, read_groups_file(str(default_groups)), quiet=True
            )
            if units:
                groups_file = str(default_groups)
            else:
                print(
                    f"NOTE: {default_groups} had no matching groups for {folder}; "
                    "falling back to auto-windowing."
                )
        if not units:
            entries = auto_entries(
                docs, args.group_size, args.overlap, args.max_page_gap
            )
            units = build_units(docs, entries)

    if not units:
        sys.exit(
            f"ERROR: no multi-page groups found in {folder}. "
            f"Run 'python make_groups.py --folder {folder}' or omit --groups-file."
        )

    max_pages = max(len(window) for _, _, window in units)
    tokens_per_image = args.max_pixels // PIXELS_PER_TOKEN
    needed = max_pages * tokens_per_image + args.max_tokens + args.text_token_reserve
    if needed > args.max_model_len:
        sys.exit(
            f"ERROR: a {max_pages}-page group needs ~{needed} tokens but "
            f"--max-model-len is {args.max_model_len}. Raise --max-model-len or "
            "lower --max-pixels / --max-tokens."
        )

    out_path = Path(args.out)
    done, seen_by_doc = load_done(out_path)
    todo = [unit for unit in units if unit[0] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"Model: {args.model} | Groups: {len(units)} | Done: {len(done)} | "
        f"Todo: {len(todo)} | Max pages/group: {max_pages} | "
        f"Groups source: {groups_file or 'auto-window'}"
    )
    if args.dry_run:
        for key, _, window in units:
            status = "done" if key in done else "todo"
            paths = [format_source_path(p, args.path_mode) for _, p in window]
            print(f"  [{status}] {key} -> {paths}")
        return
    if not todo:
        print("Nothing left to process.")
        return

    llm, processor, stops = load_model(args, max_pages)

    ok = fail = 0
    start_time = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as stream:
        for pos in range(0, len(todo), args.batch_size):
            batch_units = todo[pos : pos + args.batch_size]
            jobs = []
            for key, doc, window in batch_units:
                source_paths = [
                    format_source_path(path, args.path_mode) for _, path in window
                ]
                jobs.append(
                    {
                        "key": key,
                        "doc": doc,
                        "window": window,
                        "pages": [page for page, _ in window],
                        "source_paths": source_paths,
                        "images": {
                            page: load_image(path, args.max_edge, args.max_pixels)
                            for page, path in window
                        },
                        "schema": build_schema(source_paths),
                        "accepted": None,
                        "duplicates": [],
                        "feedback": "",
                        "raw": "",
                        "attempt": 0,
                    }
                )

            for attempt in range(1, args.attempts + 1):
                active = [job for job in jobs if job["accepted"] is None]
                if not active:
                    break
                is_final = attempt == args.attempts
                temp = 0.0 if attempt == 1 else min(0.6, 0.15 * attempt)

                requests = [
                    (
                        build_request(processor, job, seen_by_doc[job["doc"]]),
                        args.max_tokens,
                        temp,
                        job["schema"],
                    )
                    for job in active
                ]
                outputs = generate_batch(llm, stops, requests)

                for job, (text, finish_reason) in zip(active, outputs):
                    job["attempt"] = attempt
                    job["raw"] = text
                    try:
                        data = parse_json(text)
                    except ValueError as exc:
                        suffix = (
                            " (output hit max_tokens limit; keep answers concise)"
                            if finish_reason == "length"
                            else ""
                        )
                        job["feedback"] = f"{exc}{suffix}"
                        continue

                    error, cleaned, duplicates = validate_result(
                        data,
                        job,
                        seen_by_doc[job["doc"]],
                        args.dup_threshold,
                        args.require_multipage,
                        is_final,
                    )
                    if error:
                        job["feedback"] = error
                        continue

                    job["accepted"] = cleaned
                    job["duplicates"] = duplicates
                    seen_by_doc[job["doc"]].extend(
                        normalize_text(item["query"]) for item in cleaned
                    )
                    write_record(
                        stream,
                        job,
                        attempt,
                        cleaned,
                        duplicates,
                        None,
                        args.raw_json_only,
                    )
                    ok += 1
                    cats = ", ".join(item["category"] for item in cleaned)
                    print(f"OK   {job['key']} (attempt {attempt}): {cats}")

            for job in jobs:
                if job["accepted"] is None:
                    fail += 1
                    write_record(
                        stream,
                        job,
                        job["attempt"] or args.attempts,
                        None,
                        [],
                        job["feedback"] or "generation failed",
                        args.raw_json_only,
                    )
                    print(f"FAIL {job['key']}: {job['feedback']}")

            processed = min(pos + args.batch_size, len(todo))
            elapsed = time.time() - start_time
            print(
                f"[{processed}/{len(todo)}] ok={ok} fail={fail} | "
                f"{elapsed / processed:.1f}s per group"
            )

    print(
        f"\nDONE  OK: {ok} | Failed: {fail} | "
        f"Time: {time.time() - start_time:.0f}s | Output: {out_path}"
    )


if __name__ == "__main__":
    main()
