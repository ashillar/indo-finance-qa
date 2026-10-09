"""Generate multi-page QA pairs from page images with InternVL and vLLM.

Standalone version of the batch script. Every question must combine two
different pages of its window:

* each slot (q1, q2, q3) is given its own pair of pages, and its schema fixes
  "source_pages" to that pair;
* each question carries one evidence quote per page, filled BEFORE the answer,
  and the two quotes must differ;
* near-duplicate questions are rejected (not just labelled) and retried with
  the rejection reason in the prompt.

Windows come from --groups-file (one document per line, "name: 38-39 45-47")
or, without it, from automatic windows over nearby pages.

    python multi_internvl_fixed.py --folder bi_images/multi \\
        --groups-file groups.txt --dry-run
"""

import argparse
import difflib
import itertools
import json
import os
import re
import sys
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

from PIL import Image

# V100 / vLLM
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_V1", "0")

# vLLM and transformers are imported lazily so --dry-run works without a GPU.

DEFAULT_MODEL = "OpenGVLab/InternVL2_5-8B"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CATEGORIES = [
    "Text-Text Inference",
    "Chart-Information Extraction",
    "Chart-Numerical Calculation",
    "Chart-Time-Sensitive",
    "Table-Numerical Calculation",
    "Table-Comparison and Sorting",
]
SLOTS = ("q1", "q2", "q3")
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)
BANNED_PHRASES = (
    "tabel ini",
    "tabel di atas",
    "grafik ini",
    "gambar ini",
    "teks ini",
    "teks di atas",
    "dokumen ini",
    "halaman ini",
    "berdasarkan tabel",
    "berdasarkan grafik",
    "menurut data di atas",
)
MIN_QUOTE_CHARS = 8
MAX_QUOTE_SIMILARITY = 0.9

INSTRUCTIONS = """
Anda diberikan {n_pages} gambar halaman dari SATU dokumen yang sama. Setiap
gambar diberi label [Halaman N]. Nomor halaman yang sah hanya: {page_list}.
Gunakan nomor label tersebut, BUKAN nomor yang tercetak di gambar.

Buat TEPAT TIGA pertanyaan dan jawaban dalam Bahasa Indonesia, diisi pada
slot q1, q2, dan q3.

SYARAT LINTAS HALAMAN (paling penting): SETIAP pertanyaan WAJIB memakai
informasi dari DUA halaman berbeda dan TIDAK BOLEH bisa dijawab hanya dari
satu halaman. Setiap halaman harus menyumbang informasi yang BERBEDA (angka
yang sama di dua halaman tidak dihitung). Pola yang baik: membandingkan atau
menghitung selisih/rasio angka dari dua halaman; mengaitkan penjelasan teks di
satu halaman dengan data tabel/grafik di halaman lain. Pertanyaan harus
menyebut kedua informasi tersebut.

ATURAN LAIN:
1. Setiap pertanyaan memakai kategori yang BERBEDA.
2. Tanyakan fakta nyata dari isi dokumen, bukan elemen visual atau struktur.
3. Jangan gunakan: "tabel ini", "tabel di atas", "grafik ini", "gambar ini",
   "teks ini", "teks di atas", "dokumen ini", "halaman ini",
   "berdasarkan tabel", "berdasarkan grafik", "menurut data di atas",
   nomor tabel, atau nomor gambar.
4. Sebutkan wilayah, periode, sektor, indikator, atau institusi yang dibahas.
5. Jangan mengarang angka. Salin angka persis seperti terlihat; perhitungan
   hanya memakai angka yang terlihat, tulis langkah singkatnya di jawaban.
6. "answer_type": "pendek" untuk jawaban singkat, "panjang" bila perlu
   penjelasan.
7. Jangan membuat pertanyaan yang isinya hampir sama.

URUTAN PENGISIAN: "evidence" lebih dulu (kutipan singkat dan persis dari
SETIAP halaman yang ditentukan, termasuk angka beserta nama baris/kolom atau
periodenya; kedua kutipan harus berisi informasi berbeda), baru "answer"
berdasarkan evidence tersebut.

HALAMAN PER PERTANYAAN:
{pair_block}

Kategori:
{category_block}

{avoid_block}
{feedback_block}
Keluarkan HANYA JSON valid. Jangan Markdown atau ```json.

Format WAJIB:
{{
  "result": {{
    "q1": {{"query": "...", "category": "...",
           "evidence": {{{evidence_example}}},
           "answer": "...", "answer_type": "pendek",
           "source_pages": [{example_pages}]}},
    "q2": {{ ...format yang sama, dengan halaman untuk q2... }},
    "q3": {{ ...format yang sama, dengan halaman untuk q3... }}
  }}
}}
""".strip()


def normalize_name(name):
    """Repair names where percent encoding lost its percent sign."""
    if name.count(" 20") >= 2:
        name = name.replace(" 20", " ")
    return re.sub(r"\s+", " ", urllib.parse.unquote(name)).strip()


def parse_page_file(path):
    match = PAGE_PATTERN.match(normalize_name(path.stem))
    if not match:
        return None
    doc = match.group("doc").strip().rstrip("_").strip()
    return doc, int(match.group("page"))


def group_documents(images):
    groups = defaultdict(list)
    skipped = []
    for path in images:
        parsed = parse_page_file(path)
        if parsed is None:
            skipped.append(path)
            continue
        doc, page = parsed
        groups[doc].append((page, path))
    for pages in groups.values():
        pages.sort(key=lambda item: item[0])
    return dict(groups), skipped


def split_runs(pages, max_gap):
    runs = [[pages[0]]]
    for item in pages[1:]:
        if item[0] - runs[-1][-1][0] <= max_gap:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def make_windows(pages, group_size, overlap):
    if len(pages) <= group_size:
        return [pages]
    overlap = max(0, min(overlap, group_size - 1))
    step = group_size - overlap
    windows = []
    for start in range(0, len(pages), step):
        window = pages[start : start + group_size]
        if len(window) < 2:
            window = pages[-2:]
        windows.append(window)
        if start + group_size >= len(pages):
            break
    return windows


def group_key(doc, pages):
    return f"{doc}::" + "-".join(f"p{page:03d}" for page in pages)


def build_auto_units(doc_groups, group_size, overlap, max_gap):
    units, isolated = [], 0
    for doc, pages in doc_groups.items():
        for run in split_runs(pages, max_gap):
            if len(run) < 2:
                isolated += 1
                continue
            for window in make_windows(run, group_size, overlap):
                numbers = [page for page, _ in window]
                units.append((group_key(doc, numbers), doc, window))
    return units, isolated


def parse_group_spec(spec):
    """Turn "38-39", "45-47" or "38,39,41" into a sorted list of pages."""
    spec = spec.strip()
    if re.fullmatch(r"\d+-\d+", spec):
        first, last = map(int, spec.split("-"))
        if last < first:
            raise ValueError(f"bad range {spec!r}")
        return list(range(first, last + 1))
    if re.fullmatch(r"\d+(,\d+)*", spec):
        return sorted({int(part) for part in spec.split(",")})
    raise ValueError(f"cannot read page group {spec!r}")


def read_groups_file(path):
    """Return [(doc_fragment, [spec, ...]), ...] from a groups file."""
    entries = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(
                    f"line {number}: expected '<document>: 38-39 45-47'"
                )
            doc, specs = line.rsplit(":", 1)
            specs = specs.replace(";", " ").split()
            if not doc.strip() or not specs:
                raise ValueError(f"line {number}: missing document or pages")
            entries.append((doc.strip(), specs))
    return entries


def build_units_from_file(doc_groups, entries, max_group_size):
    units, problems, seen = [], [], set()
    for fragment, specs in entries:
        matches = {
            doc: pages
            for doc, pages in doc_groups.items()
            if fragment.lower() in doc.lower()
        }
        exact = {
            doc: pages
            for doc, pages in matches.items()
            if doc.lower() == fragment.lower()
        }
        matches = exact or matches
        if not matches:
            problems.append(f"no document matches {fragment!r}")
            continue
        if len(matches) > 1:
            problems.append(
                f"{fragment!r} matches {len(matches)} documents; using all: "
                + " | ".join(matches)
            )
        for doc, pages in matches.items():
            path_of = dict(pages)
            for spec in specs:
                wanted = parse_group_spec(spec)
                label = f"{doc}: pages {spec}"
                missing = [page for page in wanted if page not in path_of]
                if missing:
                    problems.append(f"{label} skipped, missing page(s) {missing}")
                    continue
                if len(wanted) < 2:
                    problems.append(f"{label} skipped, a group needs 2+ pages")
                    continue
                if len(wanted) > max_group_size:
                    problems.append(
                        f"{label} skipped, {len(wanted)} pages exceed the "
                        f"context limit of {max_group_size} pages per window"
                    )
                    continue
                key = group_key(doc, wanted)
                if key not in seen:
                    seen.add(key)
                    window = [(page, path_of[page]) for page in wanted]
                    units.append((key, doc, window))
    return units, problems


def compute_safe_group_size(
    requested, max_model_len, max_dynamic_patch, max_tokens, text_reserve
):
    """Return a page count whose worst-case vision tokens fit the context."""
    if requested <= 0:
        raise ValueError("--group-size must be positive")
    if max_dynamic_patch < 1:
        raise ValueError("--max-dynamic-patch must be at least 1")
    if max_tokens <= 0 or max_model_len <= 0:
        raise ValueError("token limits must be positive")

    tiles = max_dynamic_patch + 1 if max_dynamic_patch > 1 else 1
    tokens_per_image = tiles * 256
    available = max_model_len - max_tokens - text_reserve
    if available < tokens_per_image:
        required = tokens_per_image + max_tokens + text_reserve
        raise ValueError(
            f"--max-model-len {max_model_len} cannot fit even one image "
            f"({tokens_per_image} vision + {max_tokens} output + "
            f"{text_reserve} text-reserve tokens). Use at least {required}, "
            "or reduce --max-dynamic-patch/--max-tokens."
        )
    return min(requested, available // tokens_per_image), tokens_per_image


def load_tokenizer_and_stops(model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, use_fast=False
    )
    stops = []
    for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"):
        try:
            token_id = tok.convert_tokens_to_ids(token)
            if (
                token_id is not None
                and token_id >= 0
                and token_id != tok.unk_token_id
            ):
                stops.append(token_id)
        except Exception:
            pass
    return tok, list(dict.fromkeys(stops))


def slot_pairs(page_numbers):
    """Give each slot its own pair of pages, spread over the window."""
    pairs = list(itertools.combinations(page_numbers, 2))
    return [pairs[(index * len(pairs)) // len(SLOTS)] for index in range(3)]


def build_group_prompt(
    tok, page_numbers, avoid_queries=(), feedback=None, categories=None
):
    image_block = "\n\n".join(
        f"[Halaman {page}]\n<image>" for page in page_numbers
    )
    avoid_block = ""
    if avoid_queries:
        examples = "\n".join(f"- {query}" for query in avoid_queries[-15:])
        avoid_block = (
            "Pertanyaan berikut sudah dipakai untuk dokumen ini. Jangan "
            "mengulang atau memparafrasakannya:\n" + examples + "\n"
        )
    feedback_block = ""
    if feedback:
        feedback_block = (
            f"PERCOBAAN SEBELUMNYA DITOLAK karena: {feedback}\n"
            "Perbaiki kesalahan tersebut.\n"
        )

    pairs = slot_pairs(page_numbers)
    forced = categories or [None] * len(SLOTS)
    pair_block = "\n".join(
        f"- {slot}: gabungkan halaman {pair[0]} DAN halaman {pair[1]} "
        f'(isi evidence "halaman_{pair[0]}" dan "halaman_{pair[1]}", '
        f"source_pages [{pair[0]}, {pair[1]}])"
        + (f". Kategori WAJIB: {category}" if category else "")
        for slot, pair, category in zip(SLOTS, pairs, forced)
    )
    first = pairs[0]
    evidence_example = ", ".join(
        f'"halaman_{page}": "<kutipan dari halaman {page}>"' for page in first
    )
    instructions = INSTRUCTIONS.format(
        n_pages=len(page_numbers),
        page_list=", ".join(map(str, page_numbers)),
        pair_block=pair_block,
        category_block="\n".join(f"- {name}" for name in CATEGORIES),
        avoid_block=avoid_block,
        feedback_block=feedback_block,
        evidence_example=evidence_example,
        example_pages=", ".join(map(str, first)),
    )
    return tok.apply_chat_template(
        [{"role": "user", "content": f"{image_block}\n\n{instructions}"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def item_schema(pair, category=None):
    category_values = CATEGORIES if category is None else [category]
    evidence_keys = [f"halaman_{page}" for page in pair]
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string", "enum": category_values},
            "evidence": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in evidence_keys},
                "required": evidence_keys,
                "additionalProperties": False,
            },
            "answer": {"type": "string"},
            "answer_type": {
                "type": "string",
                "enum": ["panjang", "pendek"],
            },
            # No minItems/uniqueItems: xgrammar cannot compile those and
            # outlines then fails with "vocabulary does not allow...".
            "source_pages": {
                "type": "array",
                "items": {"type": "integer", "enum": list(pair)},
            },
        },
        "required": [
            "query",
            "category",
            "evidence",
            "answer",
            "answer_type",
            "source_pages",
        ],
        "additionalProperties": False,
    }


def build_schema(page_numbers, categories=None):
    categories = categories or [None] * len(SLOTS)
    pairs = slot_pairs(page_numbers)
    return {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    slot: item_schema(pair, category)
                    for slot, pair, category in zip(SLOTS, pairs, categories)
                },
                "required": list(SLOTS),
                "additionalProperties": False,
            }
        },
        "required": ["result"],
        "additionalProperties": False,
    }


def load_image(path, max_edge):
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = max_edge / max(width, height)
    if scale < 1:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.LANCZOS,
        )
    return image


def extract_json(text):
    if not text or not text.strip():
        raise ValueError("Model returned empty output")
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")

    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        pass

    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {exc}") from exc
    raise ValueError("JSON output was truncated")


def normalize_qa(data):
    if not isinstance(data, dict):
        raise ValueError("Root must be an object")
    result = data.get("result", data)
    if not isinstance(result, dict):
        raise ValueError("'result' must contain q1/q2/q3")
    missing = [slot for slot in SLOTS if slot not in result]
    if missing:
        raise ValueError(f"Missing slot(s): {', '.join(missing)}")
    return {"result": [result[slot] for slot in SLOTS]}


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_qa(qa, page_numbers):
    result = qa.get("result")
    if not isinstance(result, list) or len(result) != 3:
        count = len(result) if isinstance(result, list) else 0
        return False, f"Expected 3 questions, got {count}"

    pairs = slot_pairs(page_numbers)
    categories = []
    for index, (item, pair) in enumerate(zip(result, pairs)):
        tag = f"{SLOTS[index]}"
        if not isinstance(item, dict):
            return False, f"{tag} must be an object"
        for field in (
            "query",
            "category",
            "evidence",
            "answer",
            "answer_type",
            "source_pages",
        ):
            if field not in item:
                return False, f"{tag} missing '{field}'"
        for field in ("query", "answer"):
            if not isinstance(item[field], str) or not item[field].strip():
                return False, f"{tag}.{field} is empty"
        lowered = item["query"].lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                return False, f"{tag}.query uses the banned phrase '{phrase}'"
        if item["category"] not in CATEGORIES:
            return False, f"{tag} has invalid category {item['category']}"
        if item["answer_type"] not in ("panjang", "pendek"):
            return False, f"{tag} has invalid answer_type"

        pages = item["source_pages"]
        if (
            not isinstance(pages, list)
            or any(type(page) is not int for page in pages)
            or set(pages) != set(pair)
        ):
            return (
                False,
                f"{tag}.source_pages must be exactly pages "
                f"{list(pair)}, got {pages}",
            )

        evidence = item["evidence"]
        if not isinstance(evidence, dict):
            return False, f"{tag}.evidence must be an object"
        quotes = []
        for page in pair:
            quote = evidence.get(f"halaman_{page}")
            if not isinstance(quote, str) or len(quote.strip()) < MIN_QUOTE_CHARS:
                return (
                    False,
                    f"{tag}.evidence needs a real quote from page {page}",
                )
            quotes.append(normalize_text(quote))
        if (
            difflib.SequenceMatcher(None, quotes[0], quotes[1]).ratio()
            >= MAX_QUOTE_SIMILARITY
        ):
            return (
                False,
                f"{tag}: the evidence from pages {pair[0]} and {pair[1]} is "
                "the same, so the question does not need both pages",
            )
        categories.append(item["category"])

    if len(set(categories)) != 3:
        return False, f"Duplicate categories: {categories}"
    return True, None


def plan_categories(parsed):
    """Keep the distinct categories the model chose and replace repeats."""
    if not parsed:
        return None
    used, plan = [], []
    for item in parsed.get("result", [])[:3]:
        category = item.get("category") if isinstance(item, dict) else None
        if category in CATEGORIES and category not in used:
            used.append(category)
            plan.append(category)
        else:
            plan.append(None)
    plan += [None] * (3 - len(plan))
    spare = [name for name in CATEGORIES if name not in used]
    return [name or spare.pop(0) for name in plan]


def drop_evidence(qa):
    return {
        "result": [
            {key: value for key, value in item.items() if key != "evidence"}
            for item in qa["result"]
        ]
    }


def normalize_query(query):
    return re.sub(r"\s+", " ", query.strip().lower())


def is_near_duplicate(query, seen_queries, threshold=0.85):
    normalized = normalize_query(query)
    return any(
        difflib.SequenceMatcher(None, normalized, previous).ratio() >= threshold
        for previous in seen_queries
    )


def find_duplicates(queries, previous, threshold):
    """Return the queries that repeat an earlier one or each other."""
    duplicates, novel = [], []
    for query in queries:
        if is_near_duplicate(query, previous + novel, threshold):
            duplicates.append(query)
        else:
            novel.append(normalize_query(query))
    return duplicates, novel


def load_records(path):
    done, failed = set(), set()
    seen = defaultdict(list)
    if not path.exists():
        return done, failed, seen
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("group_key")
            if not key:
                continue
            if record.get("qa") is not None and not record.get("error"):
                done.add(key)
                doc = record.get("document")
                if doc:
                    for item in record["qa"].get("result", []):
                        if isinstance(item, dict) and item.get("query"):
                            seen[doc].append(normalize_query(item["query"]))
            else:
                failed.add(key)
    return done, failed - done, seen


def compact_jsonl(path):
    if not path.exists():
        return 0
    best = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("group_key")
            if not key:
                continue
            good = record.get("qa") is not None and not record.get("error")
            previous = best.get(key)
            previous_good = previous and (
                previous.get("qa") is not None and not previous.get("error")
            )
            if previous is None or good or not previous_good:
                best[key] = record
    with path.open("w", encoding="utf-8") as stream:
        for key in sorted(best):
            stream.write(json.dumps(best[key], ensure_ascii=False) + "\n")
    return len(best)


def sampling_params(max_tokens, stop_ids, temperature, schema=None):
    from vllm import SamplingParams

    common = {
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "max_tokens": max_tokens,
    }
    if stop_ids:
        common["stop_token_ids"] = stop_ids
    if schema is None:
        return SamplingParams(**common)
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **common,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
    except (ImportError, TypeError):
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            **common,
            guided_decoding=GuidedDecodingParams(json=schema),
        )
    except (ImportError, TypeError):
        print("WARNING: structured decoding unavailable")
        return SamplingParams(**common)


def is_schema_compile_error(exc):
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "vocabulary does not allow",
            "does not support advanced json schema",
            "failed to compile",
            "json schema",
        )
    )


def make_job(key, doc, pages):
    return {
        "key": key,
        "doc": doc,
        "pages": pages,
        "temperature": 0.0,
        "categories": None,
        "feedback": None,
    }


def run_batch(llm, tok, jobs, max_edge, max_tokens, stops, seen_by_doc):
    """Generate jobs and report requests silently dropped by vLLM."""
    inputs, params, submitted, errors = [], [], [], []

    for job in jobs:
        page_numbers = [page for page, _ in job["pages"]]
        try:
            images = [load_image(path, max_edge) for _, path in job["pages"]]
        except Exception as exc:
            errors.append((job, f"image load failed: {exc}"))
            continue
        inputs.append(
            {
                "prompt": build_group_prompt(
                    tok,
                    page_numbers,
                    seen_by_doc[job["doc"]],
                    job["feedback"],
                    job["categories"],
                ),
                "multi_modal_data": {"image": images},
            }
        )
        params.append(
            sampling_params(
                max_tokens,
                stops,
                job["temperature"],
                build_schema(page_numbers, job["categories"]),
            )
        )
        submitted.append(job)

    if not inputs:
        return [], errors
    try:
        outputs = llm.generate(inputs, params)
    except Exception as exc:
        if not is_schema_compile_error(exc):
            return [], errors + [
                (job, f"generation failed: {exc}") for job in submitted
            ]
        print(
            "WARNING: JSON schema failed to compile; retrying this batch "
            "without constrained decoding"
        )
        unconstrained = [
            sampling_params(max_tokens, stops, job["temperature"])
            for job in submitted
        ]
        try:
            outputs = llm.generate(inputs, unconstrained)
        except Exception as retry_exc:
            return [], errors + [
                (job, f"generation failed: {retry_exc}") for job in submitted
            ]

    paired = min(len(outputs), len(submitted))
    results = list(zip(submitted[:paired], outputs[:paired]))
    for job in submitted[paired:]:
        errors.append(
            (
                job,
                "vLLM dropped the request without returning an output; "
                "check the context/image-token budget",
            )
        )
    return results, errors


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def base_record(job, attempts):
    return {
        "group_key": job["key"],
        "document": job["doc"],
        "pages": [page for page, _ in job["pages"]],
        "source_paths": [str(path.resolve()) for _, path in job["pages"]],
        "attempts": attempts,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default="bi_images/multi")
    parser.add_argument("--out", default="bi_multipage_internvl_v2.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--group-overlap", type=int, default=1)
    parser.add_argument(
        "--max-page-gap",
        type=int,
        default=2,
        help="Pages further apart than this are never put in one window",
    )
    parser.add_argument(
        "--groups-file",
        help=(
            "Text file of page groups, one document per line: "
            "'<document name or part of it>: 38-39 45-47'. Lines starting "
            "with # are ignored. Overrides automatic windowing."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the page windows and exit without loading the model",
    )
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument("--max-dynamic-patch", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--text-token-reserve", type=int, default=1536)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--dup-threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--keep-evidence",
        action="store_true",
        help="Save the evidence quotes in the output (dropped by default)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"ERROR: folder does not exist: {folder}")
    images = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"ERROR: no images found in {folder}")

    try:
        safe_group_size, tokens_per_image = compute_safe_group_size(
            args.group_size,
            args.max_model_len,
            args.max_dynamic_patch,
            args.max_tokens,
            args.text_token_reserve,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if safe_group_size < 2:
        sys.exit(
            "ERROR: the context fits only one page per window, so multi-page "
            "QA is impossible. Lower --max-dynamic-patch or --max-tokens, or "
            "raise --max-model-len."
        )
    if safe_group_size < args.group_size:
        print(
            f"WARNING: clamping --group-size {args.group_size} to "
            f"{safe_group_size}: each image can use {tokens_per_image} "
            f"tokens and --max-model-len is {args.max_model_len}."
        )
    overlap = min(args.group_overlap, safe_group_size - 1)

    doc_groups, skipped = group_documents(images)
    if skipped:
        print(
            f"WARNING: skipping {len(skipped)} filename(s) that do not "
            "match '<document>_pNNN'."
        )

    if args.groups_file:
        try:
            entries = read_groups_file(args.groups_file)
        except (OSError, ValueError) as exc:
            sys.exit(f"ERROR: --groups-file: {exc}")
        units, problems = build_units_from_file(
            doc_groups, entries, safe_group_size
        )
        for problem in problems:
            print(f"WARNING: {problem}")
    else:
        units, isolated = build_auto_units(
            doc_groups, safe_group_size, overlap, args.max_page_gap
        )
        if isolated:
            print(
                f"NOTE: skipping {isolated} isolated page(s) with no "
                f"neighbour within {args.max_page_gap} pages."
            )

    output_path = Path(args.out)
    done, failed_before, seen_by_doc = load_records(output_path)
    skip = done | failed_before if args.skip_failed else done
    todo = [unit for unit in units if unit[0] not in skip]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]

    print(
        f"\nModel: {args.model}\nDocuments: {len(doc_groups)} | "
        f"Windows: {len(units)} | Done: {len(done)} | Todo: {len(todo)}\n"
        f"Batch: {args.batch_size} | Group size: {safe_group_size} | "
        f"Vision tokens/image: <= {tokens_per_image}\n"
    )
    if args.dry_run:
        print("Windows (dry run):")
        for key, _, window in units:
            status = "done" if key in done else "todo"
            pages = ", ".join(str(page) for page, _ in window)
            print(f"  [{status}] {key}  ->  pages {pages}")
        return
    if not todo:
        print("Nothing to process.")
        if args.compact:
            print(f"Compacted to {compact_jsonl(output_path)} records.")
        return

    from vllm import LLM

    print(f"Loading {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        limit_mm_per_prompt={"image": safe_group_size},
        trust_remote_code=True,
        tokenizer_mode="slow",
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=False,
        mm_processor_kwargs={"max_dynamic_patch": args.max_dynamic_patch},
    )
    tok, stops = load_tokenizer_and_stops(args.model)

    ok = 0
    fail = 0
    start_time = time.time()
    with output_path.open("a", encoding="utf-8") as stream:
        for position in range(0, len(todo), args.batch_size):
            chunk = todo[position : position + args.batch_size]
            jobs = [make_job(key, doc, pages) for key, doc, pages in chunk]
            last_error = {}
            last_raw = {}

            for attempt in range(1, args.attempts + 1):
                if not jobs:
                    break
                if attempt > 1:
                    print(
                        f"  retry {attempt}/{args.attempts} for "
                        f"{len(jobs)} window(s)"
                    )
                results, errors = run_batch(
                    llm,
                    tok,
                    jobs,
                    args.max_edge,
                    args.max_tokens,
                    stops,
                    seen_by_doc,
                )

                retry = []

                def reject(job, message, parsed=None):
                    last_error[job["key"]] = message
                    if "Duplicate categories" in message:
                        job["categories"] = plan_categories(parsed)
                    job["feedback"] = message
                    job["temperature"] = 0.5
                    retry.append(job)

                for job, message in errors:
                    reject(job, message)

                for job, output in results:
                    key, doc = job["key"], job["doc"]
                    page_numbers = [page for page, _ in job["pages"]]
                    completion = output.outputs[0] if output.outputs else None
                    raw = completion.text.strip() if completion else ""
                    last_raw[key] = raw
                    parsed = None
                    try:
                        parsed = normalize_qa(extract_json(raw))
                        valid, error = validate_qa(parsed, page_numbers)
                        if not valid:
                            raise ValueError(error)
                    except Exception as exc:
                        message = str(exc)
                        if completion and completion.finish_reason == "length":
                            message += (
                                f" (hit {args.max_tokens}-token output limit)"
                            )
                        reject(job, message, parsed)
                        continue

                    queries = [item["query"] for item in parsed["result"]]
                    duplicates, novel = find_duplicates(
                        queries, seen_by_doc[doc], args.dup_threshold
                    )
                    if duplicates:
                        reject(
                            job,
                            "Near-duplicate question(s): "
                            + " | ".join(duplicates),
                            parsed,
                        )
                        continue

                    seen_by_doc[doc].extend(novel)
                    last_error.pop(key, None)
                    ok += 1
                    record = base_record(job, attempt)
                    record["qa"] = (
                        parsed if args.keep_evidence else drop_evidence(parsed)
                    )
                    write_record(stream, record)
                    print(f"OK   {key}: 3/3 QA")

                jobs = retry

            for job in jobs:
                fail += 1
                message = last_error.get(job["key"], "unknown failure")
                record = base_record(job, args.attempts)
                record.update(
                    {
                        "qa": None,
                        "error": message,
                        "raw_output": last_raw.get(job["key"], ""),
                    }
                )
                write_record(stream, record)
                print(f"FAIL {job['key']}: {message}")

            elapsed = time.time() - start_time
            processed = min(position + len(chunk), len(todo))
            print(
                f"[{processed}/{len(todo)}] ok={ok} fail={fail} | "
                f"{processed / max(elapsed, 1e-9):.2f} windows/s"
            )

    elapsed = time.time() - start_time
    if args.compact:
        print(f"Compacted to {compact_jsonl(output_path)} records.")
    print(
        f"\n{'=' * 60}\nDONE\nOK: {ok} | Failed: {fail} | "
        f"Total windows: {len(todo)}\nTime: {elapsed:.1f}s | "
        f"Speed: {len(todo) / max(elapsed, 1e-9):.2f} windows/s\n"
        f"Output: {output_path}\n{'=' * 60}"
    )


if __name__ == "__main__":
    main()
