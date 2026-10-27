"""Generate multi-page QA pairs from document page images with Gemma 3 and vLLM.

Usage:
    python multi_gemma.py --folder bi_multi --groups-file bi_multi_groups.txt --out bi_multipage_gemma3_4b.jsonl
    python multi_gemma.py --folder ojk_multi --groups-file ojk_multi_groups.txt --out ojk_multipage_gemma3_4b.jsonl
    python multi_gemma.py --folder bi_multi --dry-run
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

# Quiet noisy third-party startup logs and broken vLLM cpuinfo telemetry by default
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TORCH_LOGS", "-dynamo")

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_OUT = "bi_multipage_gemma3_4b.jsonl"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)
# Gemma 3 vision encoder: 256 tokens per 896x896 image (plus pan-and-scan crops if enabled).
TOKENS_PER_IMAGE = 256

CATEGORIES = [
    "Text-Text Inference",
    "Chart-Information Extraction",
    "Chart-Numerical Calculation",
    "Chart-Time-Sensitive",
    "Table-Numerical Calculation",
    "Table-Comparison and Sorting",
]
NON_TEXT_CATEGORIES = [
    c for c in CATEGORIES if c != "Text-Text Inference"
]
TABLE_CATEGORIES = [
    "Table-Numerical Calculation",
    "Table-Comparison and Sorting",
]
CHART_CATEGORIES = [
    "Chart-Information Extraction",
    "Chart-Numerical Calculation",
    "Chart-Time-Sensitive",
]
ANSWER_TYPES = ["short", "long"]
ITEM_KEYS = ("query", "category", "answer", "answer_type", "source_path")

# Reject explicit references to table/figure/page/image numbers in queries and answers.
FIG_TABLE_NUM_RE = re.compile(
    r"\b(?:tabel|grafik|gambar|bagan|figure|table|chart)\s+"
    r"(?:no\.?\s*)?(?:\d+|[ivxlcdm]+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
PAGE_REF_RE = re.compile(
    r"\b(?:halaman|hal\.?|page|image|lembar)\s*"
    r"(?:ke[-\s]*)?(?:\d+|pertama|kedua|ketiga|keempat|[ab])\b",
    re.IGNORECASE,
)
PLACEHOLDER_ANS_RE = re.compile(
    r"^(?:jawaban\b|berikut\s+adalah\s+jawaban)|"
    r"\b(?:fakta/angka\s+dari|halaman\s+(?:pertama|kedua|[ab])\b|beserta\s+persisannya)",
    re.IGNORECASE,
)
YES_NO_QUERY_RE = re.compile(
    r"(?:^\s*apakah\b|,\s*apakah\b|\bdan\s+apakah\b)",
    re.IGNORECASE,
)
CALC_QUERY_WORD_RE = re.compile(
    r"\b(?:berapa|berapakah|hitung|hitunglah|selisih|total|jumlah|rasio|perbandingan|persentase|rata-rata)\b",
    re.IGNORECASE,
)
LEAKED_VALUE_RE = re.compile(
    r"\b(?:sebesar|senilai|mencapai|tercatat)\s+(?:rp\.?\s*)?-?\d+(?:[.,]\d+)?\s*(?:%|persen|triliun|miliar|juta|ribu)?",
    re.IGNORECASE,
)

CATEGORY_EXAMPLES = """
Examples (from a DIFFERENT report, "Analisis Uang Beredar Agustus 2022"; illustrate style only — NEVER copy these numbers/topics, use ONLY facts visible in the current images):
- [Text-Text Inference]
  evidence_page_1: "pangsa tabungan rupiah ditarik sewaktu-waktu 48,7% terhadap M1 (narasi komposisi M1)"
  evidence_page_2: "tagihan bersih sistem moneter kepada Pemerintah Pusat kontraksi 18,3% (yoy)"
  query: "Pada dokumen Analisis Uang Beredar Agustus 2022, bagaimana peran pangsa tabungan rupiah yang dapat ditarik sewaktu-waktu dalam komposisi M1 serta bagaimana perkembangan tagihan bersih sistem moneter kepada Pemerintah Pusat pada Agustus 2022?"
  answer: "Dalam komposisi M1 Agustus 2022, tabungan rupiah yang dapat ditarik sewaktu-waktu memiliki pangsa sebesar 48,7% terhadap total M1, sementara dari sisi faktor yang memengaruhi M2, tagihan bersih sistem moneter kepada Pemerintah Pusat mengalami kontraksi sebesar 18,3% (yoy)." (long)
- [Table-Numerical Calculation]
  evidence_page_1: "pangsa uang kartal di luar bank umum dan BPR 20,0% terhadap M1"
  evidence_page_2: "pangsa uang kuasi 43,5% terhadap M2 (Rp3.434,1 triliun)"
  query: "Dalam laporan Analisis Uang Beredar Agustus 2022, berapakah selisih antara pangsa uang kuasi terhadap M2 dengan pangsa uang kartal di luar bank umum dan BPR terhadap M1 pada Agustus 2022?"
  answer: "Pangsa uang kuasi terhadap M2 tercatat sebesar 43,5% dan pangsa uang kartal di luar bank umum dan BPR terhadap M1 sebesar 20,0%, sehingga selisihnya adalah 43,5% - 20,0% = 23,5 poin persentase." (short)
"""

PROMPT_TEMPLATE = """Document: {doc_name}
Pages in this group:
{source_list}

Generate THREE strictly multi-page QA pairs (`q1`, `q2`, `q3`) in Bahasa Indonesia using ONLY facts visible in the provided page images.

CRITICAL RULES:
1. TRUE MULTI-PAGE ONLY (NO SINGLE-PAGE LEAKS):
   - Each question MUST combine a specific fact/number that appears ONLY on its first assigned page (`evidence_page_1`, max 20 words) and a DIFFERENT specific fact/number that appears ONLY on its second assigned page (`evidence_page_2`, max 20 words).
   - NEVER pick two facts that both appear on the same page! Specifically, if Page B has a summary table (`Tabel`) that repeats headline numbers from Page A's narrative text, do NOT pair a narrative number from Page A with a table number that is ALSO in that same table on Page B (or vice versa). Pick a detail/chart/sub-component unique to Page A (absent from Page B) and pair it with a table row/chart/detail unique to Page B (absent from Page A).
   - The `query` itself MUST explicitly ask about BOTH `evidence_page_1` (from the 1st page) AND `evidence_page_2` (from the 2nd page) so that someone looking at only one page CANNOT answer the question.
2. Distinct topics across `q1`, `q2`, `q3`:
   - `q1`, `q2`, and `q3` MUST ask about THREE COMPLETELY DIFFERENT indicators, table rows, or chart series, and must NOT reuse the same numbers across slots.
3. Open questions only — NO "Apakah" yes/no questions and NO leaking target numbers in the query:
   - Start queries with open question words (`Bagaimana`, `Mengapa`, `Apa`, `Berapakah`, `Berapa`, `Komponen/Daerah/Sektor manakah`). NEVER use `"Apakah ..."` confirmation questions.
   - For calculation questions (`Table-Numerical Calculation`, `Chart-Numerical Calculation`), ask for the difference/sum/ratio by naming the TWO distinct indicators from the two different pages — do NOT give away the numbers inside the `query` itself. State both numbers and show the arithmetic in `answer`.
4. Match visual type to category: Choose `Table-*` ONLY when extracting from a table (`Tabel`), and `Chart-*` ONLY when extracting from a chart/graph (`Grafik`/`Bagan`). Never mention page numbers (`halaman 70`), image numbers, or table/figure numbers (`Tabel 2.10`, `Grafik 1`) in `query` or `answer`.
5. Write REAL answers with exact numbers and facts from both pages — NEVER write placeholder descriptions like `"Jawaban yang menyebutkan..."`.
6. For each slot (`q1`, `q2`, `q3`), fill fields in this exact order:
   - `evidence_page_1`: Short exact fact + number unique to the first assigned page and NOT present on the second assigned page (<= 20 words).
   - `evidence_page_2`: Different short exact fact + number unique to the second assigned page and NOT present on the first assigned page (<= 20 words).
   - `query`: Open question in Bahasa Indonesia naming "{doc_name}" that explicitly asks about BOTH the indicator in `evidence_page_1` AND the indicator in `evidence_page_2`.
   - `category`: Must match one of the slot's allowed categories and match the actual visual element (table vs chart).
   - `answer`: Complete, accurate answer in Bahasa Indonesia explicitly citing the exact facts/numbers from BOTH `evidence_page_1` and `evidence_page_2` (plus the calculation step for numerical categories).
   - `answer_type`: `"short"` (concise) or `"long"` (detailed explanation / multi-step calculation).
   - `source_path`: Exact 2-element array of file paths for the slot's assigned page pair.

Assigned Slots:
{slot_instructions}

Categories:
- Text-Text Inference: Combine textual explanations/causes across both pages.
- Chart-Information Extraction: Extract and compare chart values across pages.
- Chart-Numerical Calculation: Compute differences, totals, or ratios from chart values across pages.
- Chart-Time-Sensitive: Compare multi-period chart trends across pages.
- Table-Numerical Calculation: Compute differences, totals, shares, or growth combining a value unique to page 1 and a table value unique to page 2.
- Table-Comparison and Sorting: Rank or compare entities using values from both pages.
{category_examples}{feedback_block}
CRITICAL: Return ONLY the raw JSON object starting with `{{` and ending with `}}` (no Markdown fences or extra text):
{{"q1": {{"evidence_page_1": "<fact unique to 1st page>", "evidence_page_2": "<fact unique to 2nd page>", "query": "<open question asking about both>", "category": "Text-Text Inference", "answer": "<real answer citing both pages>", "answer_type": "long", "source_path": ["...", "..."]}}, "q2": {{"evidence_page_1": "<number unique to 1st page>", "evidence_page_2": "<number unique to 2nd page>", "query": "<calculation question across both pages>", "category": "Table-Numerical Calculation", "answer": "<values + calculation result>", "answer_type": "short", "source_path": ["...", "..."]}}, "q3": {{"evidence_page_1": "<value unique to 1st page>", "evidence_page_2": "<value unique to 2nd page>", "query": "<comparison/ranking question across both pages>", "category": "Table-Comparison and Sorting", "answer": "<comparison answer citing both values>", "answer_type": "short", "source_path": ["...", "..."]}}}}"""

PROBE_PROMPT = """Gambar di atas adalah SATU halaman dokumen. Periksa apakah fakta/informasi berikut terlihat jelas di halaman ini saja, dan apakah pertanyaan dapat dijawab LENGKAP hanya dari halaman ini:
- Fakta A: {ev1}
- Fakta B: {ev2}
- Pertanyaan: {query}

Aturan:
1. "ev1_on_page": true jika angka/fakta spesifik pada Fakta A tertulis di halaman ini, false jika tidak ada di halaman ini.
2. "ev2_on_page": true jika angka/fakta spesifik pada Fakta B tertulis di halaman ini, false jika tidak ada di halaman ini.
3. "answerable": true HANYA jika SELURUH informasi yang ditanyakan pada Pertanyaan tersedia lengkap di halaman ini saja tanpa perlu melihat halaman lain.
4. "answer": jawaban lengkap berdasarkan halaman ini saja (kosongkan "" jika tidak bisa dijawab lengkap dari halaman ini).

Keluarkan HANYA JSON valid: {{"ev1_on_page": true/false, "ev2_on_page": true/false, "answerable": true/false, "answer": "..."}}"""

PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "ev1_on_page": {"type": "boolean"},
        "ev2_on_page": {"type": "boolean"},
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["ev1_on_page", "ev2_on_page", "answerable", "answer"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Helpers: text, JSON, paths, and page grouping
# --------------------------------------------------------------------------


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def strip_doc_boilerplate(query, doc_name=""):
    """Strip common document-title prefixes and boilerplate phrases before comparing queries."""
    text = normalize_text(query)
    if doc_name:
        doc_norm = normalize_text(doc_name)
        text = text.replace(doc_norm, " ")
    text = re.sub(
        r"\b(?:pada|dalam|berdasarkan|menurut)\s+(?:dokumen|laporan)\b",
        " ",
        text,
    )
    text = re.sub(
        r"\b(?:bagaimana|apakah|berapakah|berapa)\s+(?:perbandingan|perubahan|perkembangan|perbedaan|selisih|hubungan|kaitan)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def is_near_duplicate(query, previous, threshold, doc_name=""):
    norm = strip_doc_boilerplate(query, doc_name)
    for other in previous:
        other_norm = strip_doc_boilerplate(other, doc_name)
        if norm == other_norm:
            return True
        if difflib.SequenceMatcher(None, norm, other_norm).ratio() >= threshold:
            return True
    return False


def parse_json(text, prefilled_brace=False):
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).strip()
    if prefilled_brace and not cleaned.startswith("{"):
        cleaned = "{" + cleaned
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
# Prompt & Schema Construction (Gemma 3 Format)
# --------------------------------------------------------------------------


def format_gemma_chat(processor, content_blocks, prefill_brace=False):
    """Apply Gemma 3 chat template to a list of multimodal content blocks."""
    messages = [{"role": "user", "content": content_blocks}]
    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        parts = []
        for block in content_blocks:
            if block.get("type") == "image":
                parts.append("<start_of_image>")
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
        joined = "\n".join(parts)
        prompt = f"<bos><start_of_turn>user\n{joined}<end_of_turn>\n<start_of_turn>model\n"
    if prefill_brace:
        prompt = prompt.rstrip() + "\n{"
    return prompt


def slot_pairs(pages):
    """Assign a page pair (page_a, page_b) for each of the 3 questions (q1, q2, q3)."""
    if len(pages) < 2:
        return [(pages[0], pages[0])] * 3
    adjacent = list(zip(pages, pages[1:]))
    others = [(a, b) for i, a in enumerate(pages) for b in pages[i + 2 :]]
    pairs = adjacent + others
    return [pairs[i % len(pairs)] for i in range(3)]


def build_schema(source_paths, require_multipage=True):
    source_path_schema = {
        "type": "array",
        "items": {"type": "string"},
    }

    def make_item_schema(allowed_categories):
        return {
            "type": "object",
            "properties": {
                "evidence_page_1": {"type": "string", "maxLength": 200},
                "evidence_page_2": {"type": "string", "maxLength": 200},
                "query": {"type": "string"},
                "category": {"type": "string", "enum": allowed_categories},
                "answer": {"type": "string"},
                "answer_type": {"type": "string", "enum": ANSWER_TYPES},
                "source_path": source_path_schema,
            },
            "required": [
                "evidence_page_1",
                "evidence_page_2",
                *ITEM_KEYS,
            ],
            "additionalProperties": False,
        }

    q1_cats = ["Text-Text Inference", "Chart-Time-Sensitive"]
    q2_cats = ["Table-Numerical Calculation", "Chart-Numerical Calculation"]
    q3_cats = ["Table-Comparison and Sorting", "Chart-Information Extraction"]
    return {
        "type": "object",
        "properties": {
            "q1": make_item_schema(q1_cats),
            "q2": make_item_schema(q2_cats),
            "q3": make_item_schema(q3_cats),
        },
        "required": ["q1", "q2", "q3"],
        "additionalProperties": False,
    }


def build_request(processor, job, avoid_queries, prefill_brace=False):
    source_paths = job["source_paths"]
    page_to_path = {
        page: sp for (page, _), sp in zip(job["window"], source_paths)
    }
    source_list = "\n".join(
        f"- [Page {page}]: `{sp}`"
        for (page, _), sp in zip(job["window"], source_paths)
    )
    example_multi = ", ".join(f'"{p}"' for p in source_paths[:2])

    pairs = slot_pairs(job["pages"])
    slot_specs = [
        (
            "q1",
            pairs[0],
            "`Text-Text Inference` or `Chart-Time-Sensitive`",
            "narrative reasoning, policy/trend explanation, or time-series comparison (Topic 1)",
        ),
        (
            "q2",
            pairs[1],
            "`Table-Numerical Calculation` or `Chart-Numerical Calculation`",
            "numerical calculation (difference, sum, or ratio) on a DIFFERENT metric than q1 (Topic 2)",
        ),
        (
            "q3",
            pairs[2],
            "`Table-Comparison and Sorting` or `Chart-Information Extraction`",
            "ranking, sorting, or comparing a THIRD distinct metric/sub-component not in q1/q2 (Topic 3)",
        ),
    ]
    slot_lines = []
    for slot_name, (pa, pb), cat_desc, focus_desc in slot_specs:
        spa = page_to_path.get(pa, "")
        spb = page_to_path.get(pb, "")
        slot_lines.append(
            f"- `{slot_name}`: Combine [Page {pa}] + [Page {pb}] | category: {cat_desc} | "
            f"focus: {focus_desc} | source_path: [\"{spa}\", \"{spb}\"]"
        )
    slot_instructions = "\n".join(slot_lines)

    feedback_lines = []
    locked = job.get("locked_slots") or {}
    locked_desc = [
        f"q{i} (category={locked[i]['category']!r}, query={locked[i]['query']!r})"
        for i in (1, 2, 3)
        if locked.get(i) is not None
    ]
    if locked_desc:
        feedback_lines.append(
            "LOCKED SLOTS (already verified; do NOT reuse their categories or metrics/numbers in the remaining slots): "
            + "; ".join(locked_desc)
        )
    if job["feedback"]:
        feedback_lines.append(
            f"PREVIOUS ATTEMPT REJECTED: {job['feedback']}\n"
            "Fix all reported issues on the unlocked slot(s) and regenerate the JSON output."
        )
    feedback_block = ("\n" + "\n".join(feedback_lines) + "\n") if feedback_lines else ""

    prompt_text = PROMPT_TEMPLATE.format(
        doc_name=job["doc"],
        source_list=source_list,
        slot_instructions=slot_instructions,
        feedback_block=feedback_block,
        example_multi_paths=example_multi,
        category_examples=CATEGORY_EXAMPLES,
    )

    content = []
    for (page, _), sp in zip(job["window"], source_paths):
        content.append(
            {
                "type": "text",
                "text": f"[START OF PAGE {page} | source_path: {sp}]",
            }
        )
        content.append({"type": "image"})
        content.append(
            {
                "type": "text",
                "text": f"[END OF PAGE {page}]",
            }
        )
    content.append({"type": "text", "text": prompt_text})

    prompt = format_gemma_chat(processor, content, prefill_brace=prefill_brace)
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
    cand_base = os.path.basename(candidate)
    cand_stem = Path(cand_base).stem
    norm_cand = normalize_name(cand_stem).lower()
    for vp in valid_paths:
        if os.path.basename(vp) == cand_base:
            return vp
        if normalize_name(Path(vp).stem).lower() == norm_cand:
            return vp
    page_match = re.search(r"p?0*(\d+)$", cand_stem, re.IGNORECASE)
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
    for item in raw_list:
        matched = resolve_one_source_path(item, valid_paths, pages_to_path)
        if matched is None:
            return None
        if matched not in resolved:
            resolved.append(matched)

    if not resolved:
        return None
    order = {vp: idx for idx, vp in enumerate(valid_paths)}
    resolved.sort(key=lambda p: order.get(p, 999))
    return resolved[0] if len(resolved) == 1 else resolved


def normalize_answer_type(raw_type, answer):
    val = str(raw_type or "").strip().lower()
    if val in ANSWER_TYPES:
        return val
    return "long" if len(answer.split()) > 18 else "short"


def strip_internal_keys(item):
    return {k: item[k] for k in ITEM_KEYS}


def validate_slot(
    idx,
    item,
    job,
    seen_queries,
    dup_threshold,
    require_multipage,
    used_categories,
    batch_queries,
    used_num_sets,
):
    """Validate a single slot (`q1`, `q2`, or `q3`). Returns `(error_msg_or_None, cleaned_dict_or_None)`."""
    if not isinstance(item, dict):
        return f"q{idx} is not an object", None

    valid_paths = job["source_paths"]
    pages_to_path = {
        page: sp for (page, _), sp in zip(job["window"], valid_paths)
    }
    pairs = slot_pairs(job["pages"])
    pa, pb = pairs[idx - 1]

    ev1 = str(item.get("evidence_page_1") or "").strip()
    ev2 = str(item.get("evidence_page_2") or "").strip()
    query = str(item.get("query") or "").strip()
    category = str(item.get("category") or "").strip()
    answer = str(item.get("answer") or "").strip()
    raw_type = item.get("answer_type")
    raw_source = item.get("source_path")

    if require_multipage and len(valid_paths) >= 2:
        if len(ev1) < 5 or len(ev2) < 5:
            return (
                f"q{idx} must provide non-empty 'evidence_page_1' (from Page {pa}) "
                f"and 'evidence_page_2' (from Page {pb})"
            ), None
        if len(ev1) > 260 or len(ev2) > 260:
            return (
                f"q{idx} 'evidence_page_1'/'evidence_page_2' is too long; "
                "keep each evidence string concise (under 20 words, e.g. 'indicator X = Y%')"
            ), None
        norm_ev1, norm_ev2 = normalize_text(ev1), normalize_text(ev2)
        if (
            norm_ev1 in norm_ev2
            or norm_ev2 in norm_ev1
            or difflib.SequenceMatcher(None, norm_ev1, norm_ev2).ratio() >= 0.75
        ):
            return (
                f"q{idx} 'evidence_page_1' and 'evidence_page_2' are too similar; "
                f"they must extract DIFFERENT facts from Page {pa} and Page {pb}"
            ), None
        ev1_nums = extract_number_keys(ev1)
        ev2_nums = extract_number_keys(ev2)
        if ev1_nums and ev2_nums and ev1_nums == ev2_nums:
            return (
                f"q{idx} 'evidence_page_1' and 'evidence_page_2' cite the exact same number(s); "
                f"extract DIFFERENT metrics/values from Page {pa} and Page {pb}"
            ), None
        combined_qa = f"{query} {answer}"
        if not evidence_supported_in_text(
            ev1, answer, context_text=combined_qa
        ) or not evidence_supported_in_text(
            ev2, answer, context_text=combined_qa
        ):
            return (
                f"q{idx} answer does not cite both pages' evidences; the answer MUST "
                f"explicitly include the specific facts/numbers AND indicator names from BOTH "
                f"'evidence_page_1' (Page {pa}) and 'evidence_page_2' (Page {pb})"
            ), None
        if not query_combines_evidences(query, ev1, ev2):
            return (
                f"q{idx} query only asks about one page's topic; rewrite q{idx} query so the "
                f"question itself explicitly asks about BOTH the indicator in 'evidence_page_1' "
                f"(Page {pa}) AND the indicator in 'evidence_page_2' (Page {pb})"
            ), None

    if len(query) < 15:
        return f"q{idx} query is too short or empty", None
    if len(answer) < 5:
        return f"q{idx} answer is too short or empty", None
    if PLACEHOLDER_ANS_RE.search(answer):
        return (
            f"q{idx} answer is a placeholder template ({answer[:60]!r}); "
            "write the actual concrete answer with real facts and numbers from the pages"
        ), None
    if PAGE_REF_RE.search(query) or PAGE_REF_RE.search(answer):
        return (
            f"q{idx} explicitly mentions a page number ('halaman ...' / 'page ...'); "
            "do NOT mention page numbers in query or answer — name the indicator/region/period instead"
        ), None
    if FIG_TABLE_NUM_RE.search(query):
        return (
            f"q{idx} query explicitly mentions a figure or table number; "
            "describe the content instead"
        ), None
    if YES_NO_QUERY_RE.search(query):
        return (
            f"q{idx} uses a yes/no question ('Apakah...'); "
            "rewrite using 'Bagaimana', 'Berapa', 'Mengapa', or 'Komponen/daerah mana' "
            "without stating the answer's numbers inside the question"
        ), None
    if category in ("Table-Numerical Calculation", "Chart-Numerical Calculation"):
        if not CALC_QUERY_WORD_RE.search(query):
            return (
                f"q{idx} has category {category!r} but does not ask for a calculation; "
                "ask 'Berapa selisih / jumlah / total / rata-rata / rasio ...'"
            ), None
        leaked_vals = LEAKED_VALUE_RE.findall(query)
        if len(leaked_vals) >= 2:
            return (
                f"q{idx} leaks the numbers ({', '.join(leaked_vals[:3])}) inside the query; "
                "ask 'Berapa selisih/jumlah/perbandingan [metric A] dan [metric B]?' "
                "and put the numbers ONLY in the answer"
            ), None
        if len(extract_number_keys(answer)) < 2:
            return (
                f"q{idx} ({category}) answer must show the two source numbers and the calculated result"
            ), None
    if category not in CATEGORIES:
        return (
            f"q{idx} has invalid category {category!r}; "
            f"must be one of {CATEGORIES}"
        ), None
    if category in used_categories:
        return (
            f"q{idx} uses duplicate category {category!r}; all 3 questions must have "
            "distinct categories"
        ), None

    expected_pair_paths = [
        pages_to_path[p] for p in (pa, pb) if p in pages_to_path
    ]
    norm_source = normalize_source_path(raw_source, valid_paths, pages_to_path)
    if norm_source is None:
        if require_multipage and len(expected_pair_paths) >= 2:
            norm_source = expected_pair_paths
        else:
            return (
                f"q{idx} has invalid source_path {raw_source!r}; "
                f"use exact paths from {valid_paths}"
            ), None
    if require_multipage and len(valid_paths) >= 2:
        if not isinstance(norm_source, list) or len(norm_source) < 2:
            if len(expected_pair_paths) >= 2:
                norm_source = expected_pair_paths
            elif len(valid_paths) == 2:
                norm_source = list(valid_paths)
            else:
                return (
                    f"q{idx} only cites 1 page ({norm_source!r}); "
                    "every question must combine at least 2 pages in source_path"
                ), None

    doc_name = job.get("doc", "")
    for prev_q in batch_queries:
        if is_near_duplicate(query, [prev_q], dup_threshold, doc_name=doc_name):
            return (
                f"q{idx} repeats another question in this group ({query!r}); "
                f"rewrite q{idx} to ask about a completely different indicator or metric "
                f"on Page {pa} and Page {pb}"
            ), None

    slot_nums = extract_number_keys(f"{ev1} {ev2}")
    if slot_nums:
        for prev_nums in used_num_sets:
            if prev_nums and len(slot_nums & prev_nums) >= 2:
                return (
                    f"q{idx} reuses the same numbers ({sorted(slot_nums & prev_nums)}) as an earlier slot; "
                    f"pick a DIFFERENT metric/table row/chart series for q{idx}"
                ), None

    if is_near_duplicate(query, seen_queries, dup_threshold, doc_name=doc_name):
        return f"q{idx} repeats a question from an earlier group; pick a different metric", None

    answer_type = normalize_answer_type(raw_type, answer)
    used_categories.add(category)
    batch_queries.append(query)
    if slot_nums:
        used_num_sets.append(slot_nums)

    return None, {
        "query": query,
        "category": category,
        "answer": answer,
        "answer_type": answer_type,
        "source_path": norm_source,
        "_ev1": ev1,
        "_ev2": ev2,
        "_pages": (pa, pb),
        "_nums": slot_nums,
    }


def validate_candidates_per_slot(data, job, seen_queries, dup_threshold, require_multipage):
    """Validate unlocked slots individually while preserving already-locked slots on the job."""
    if not isinstance(data, dict):
        return {"all": "top-level JSON must be an object"}, {}

    if all(k in data for k in ("q1", "q2", "q3")):
        result = [data["q1"], data["q2"], data["q3"]]
    else:
        result = data.get("result")
    if not isinstance(result, list) or len(result) != 3:
        return {"all": "JSON must contain q1, q2, q3 (or 'result' list of 3 objects)"}, {}

    locked = job["locked_slots"]
    used_categories = {
        locked[i]["category"] for i in (1, 2, 3) if locked.get(i) is not None
    }
    batch_queries = [
        locked[i]["query"] for i in (1, 2, 3) if locked.get(i) is not None
    ]
    used_num_sets = [
        locked[i]["_nums"]
        for i in (1, 2, 3)
        if locked.get(i) is not None and locked[i].get("_nums")
    ]

    errors = {}
    new_candidates = {}
    for idx in (1, 2, 3):
        if locked.get(idx) is not None:
            continue
        err, cleaned = validate_slot(
            idx,
            result[idx - 1],
            job,
            seen_queries,
            dup_threshold,
            require_multipage,
            used_categories,
            batch_queries,
            used_num_sets,
        )
        if err:
            errors[idx] = err
        else:
            new_candidates[idx] = cleaned

    return errors, new_candidates


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "dalam",
    "adalah", "ini", "itu", "atau", "juga", "oleh", "sebesar", "menjadi",
    "tahun", "triwulan", "bulan", "periode", "provinsi", "kota", "kabupaten",
    "persen", "yoy", "mtm", "qtq", "terhadap", "dibandingkan", "seiring",
    "antara", "serta", "nilai", "jumlah", "tingkat", "pertumbuhan",
    "agustus", "september", "oktober", "november", "desember", "januari",
    "februari", "maret", "april", "mei", "juni", "juli", "posisi", "analisis",
    "laporan", "dokumen", "kajian", "ekonomi", "keuangan", "regional",
    "mencapai", "tercatat", "mengalami", "peningkatan", "penurunan", "tumbuh",
}


def extract_number_keys(text):
    """Extract normalized multi-digit numbers (excluding 4-digit years) for comparison."""
    keys = set()
    for token in NUMBER_RE.findall(text or ""):
        digits = re.sub(r"\D", "", token)
        if YEAR_RE.fullmatch(token) or len(digits) < 2:
            continue
        keys.add(digits.lstrip("0") or "0")
    return keys


def content_words(text):
    return {
        w for w in re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
        if w not in STOPWORDS_ID
    }


def evidence_supported_in_text(evidence, text, context_text=None):
    """Check that `text` uses the specific number AND topic words from `evidence`."""
    ev_nums = extract_number_keys(evidence)
    text_nums = extract_number_keys(text)
    ev_words = content_words(evidence)
    check_words_in = content_words(context_text if context_text is not None else text)

    if ev_nums:
        if not (ev_nums & text_nums):
            return False
        if ev_words and not (ev_words & check_words_in):
            return False
        return True

    if not ev_words:
        return False
    text_words = content_words(text)
    overlap = ev_words & text_words
    required = 1 if len(ev_words) <= 2 else 2
    return len(overlap) >= required


def query_combines_evidences(query, ev1, ev2):
    """Verify the query itself asks about both evidence_page_1 and evidence_page_2 topics."""
    q_words = content_words(query)
    w1 = content_words(ev1)
    w2 = content_words(ev2)
    if w1 and w2:
        if (w1 & q_words) and (w2 & q_words):
            return True
    return False


def probe_answers_match(reference, probe, ev1="", ev2=""):
    probe_nums = extract_number_keys(probe)
    if ev1 and ev2:
        if evidence_supported_in_text(ev1, probe) and evidence_supported_in_text(ev2, probe):
            return True
        ev1_nums = extract_number_keys(ev1)
        ev2_nums = extract_number_keys(ev2)
        if ev1_nums and ev2_nums and (ev1_nums & probe_nums) and (ev2_nums & probe_nums):
            return True
    ref_nums = extract_number_keys(reference)
    if ref_nums:
        return ref_nums <= probe_nums
    ratio = difflib.SequenceMatcher(
        None, normalize_text(reference), normalize_text(probe)
    ).ratio()
    return ratio >= 0.60


def probe_single_page_leaks(
    llm,
    processor,
    stops,
    slot_candidates,
    guided_decoding=False,
    guided_backend=None,
    prefill_brace=False,
):
    """Probe each candidate slot against each of its two pages individually.

    Rejects a slot if:
    1. Either single page already contains BOTH `evidence_page_1` and `evidence_page_2`, OR
    2. Either single page alone can answer the question, OR
    3. `evidence_page_1` is not found on Page A or `evidence_page_2` is not found on Page B.
    """
    requests, meta = [], []
    for job, idx, item in slot_candidates:
        pa, pb = item["_pages"]
        ev1, ev2 = item["_ev1"], item["_ev2"]
        for page in (pa, pb):
            if page not in job["images"]:
                continue
            probe_text = PROBE_PROMPT.format(
                ev1=ev1,
                ev2=ev2,
                query=item["query"],
            )
            content = [
                {"type": "image"},
                {"type": "text", "text": probe_text},
            ]
            prompt = format_gemma_chat(
                processor, content, prefill_brace=prefill_brace
            )
            requests.append(
                (
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": [job["images"][page]]},
                    },
                    256,
                    0.0,
                    PROBE_SCHEMA if guided_decoding else None,
                )
            )
            meta.append((job["key"], idx, page, pa, pb, ev1, ev2, item["answer"]))

    if not requests:
        return {}

    probe_results = {}
    for (key, idx, page, pa, pb, ev1, ev2, ref_ans), (text, _) in zip(
        meta, generate_batch(llm, stops, requests, backend=guided_backend)
    ):
        try:
            data = parse_json(text, prefilled_brace=prefill_brace)
        except ValueError:
            continue
        if isinstance(data, dict):
            probe_results[(key, idx, page)] = (data, pa, pb, ev1, ev2, ref_ans)

    leaks = {}
    for job, idx, item in slot_candidates:
        key = job["key"]
        pa, pb = item["_pages"]
        ev1, ev2 = item["_ev1"], item["_ev2"]
        ref_ans = item["answer"]

        res_a = probe_results.get((key, idx, pa))
        res_b = probe_results.get((key, idx, pb))

        for page, res in ((pa, res_a), (pb, res_b)):
            if res is None:
                continue
            data = res[0]
            ev1_on = bool(data.get("ev1_on_page"))
            ev2_on = bool(data.get("ev2_on_page"))
            ans_alone = bool(data.get("answerable"))
            probe_ans = str(data.get("answer") or "")

            if ev1_on and ev2_on:
                which_ev = "evidence_page_2" if page == pa else "evidence_page_1"
                target_page = pb if page == pa else pa
                leaks[(key, idx)] = (
                    f"q{idx} is NOT multi-page: Page {page} alone already contains BOTH "
                    f"'evidence_page_1' ({ev1!r}) and 'evidence_page_2' ({ev2!r}). "
                    f"Pick '{which_ev}' from a detail/chart/table that appears ONLY on Page {target_page} "
                    f"and is NOT repeated on Page {page}"
                )
                break

            if ans_alone and probe_answers_match(ref_ans, probe_ans, ev1=ev1, ev2=ev2):
                leaks[(key, idx)] = (
                    f"q{idx} can be answered from Page {page} alone; "
                    f"rewrite q{idx} using a fact unique to Page {pa} (not on Page {pb}) "
                    f"and a fact unique to Page {pb} (not on Page {pa})"
                )
                break

        if (key, idx) in leaks:
            continue

        if res_a is not None:
            data_a = res_a[0]
            probe_ans_a = str(data_a.get("answer") or "")
            if not data_a.get("ev1_on_page") and not evidence_supported_in_text(ev1, probe_ans_a):
                leaks[(key, idx)] = (
                    f"q{idx} 'evidence_page_1' ({ev1!r}) was NOT found on Page {pa}; "
                    f"extract 'evidence_page_1' strictly from Page {pa}"
                )
                continue
        if res_b is not None:
            data_b = res_b[0]
            probe_ans_b = str(data_b.get("answer") or "")
            if not data_b.get("ev2_on_page") and not evidence_supported_in_text(ev2, probe_ans_b):
                leaks[(key, idx)] = (
                    f"q{idx} 'evidence_page_2' ({ev2!r}) was NOT found on Page {pb}; "
                    f"extract 'evidence_page_2' strictly from Page {pb}"
                )
                continue

    return leaks


# --------------------------------------------------------------------------
# vLLM Execution
# --------------------------------------------------------------------------


def resolve_dtype(requested, model_name):
    """Gemma 3 overflows in fp16 (all-<pad> output). GPUs without bf16 (e.g. V100) must use fp32."""
    import torch

    has_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    is_gemma = "gemma" in model_name.lower()
    if requested is None:
        if is_gemma:
            return "bfloat16" if has_bf16 else "float32"
        return "half" if not has_bf16 else "bfloat16"
    if is_gemma and requested == "half":
        print(
            "WARNING: Gemma 3 in fp16 usually produces empty/<pad> output. "
            "Use --dtype float32 (V100) or bfloat16 (Ampere+)."
        )
    if is_gemma and requested == "bfloat16" and not has_bf16:
        print("WARNING: this GPU has no bf16 support; falling back to float32.")
        return "float32"
    return requested


def configure_vllm_env(model_name, attention_backend):
    """Pick a vLLM engine/attention backend that runs Gemma 3 sliding-window attention correctly.

    On pre-Ampere GPUs (V100) vLLM < 0.11 only offers XFormers or a FlexAttention without
    sliding-window support; XFormers silently disables
    Gemma 3's sliding window and produces repetitive garbage on prompts > 1024 tokens.
    """
    import torch
    from packaging.version import Version
    import vllm

    has_flash = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    is_gemma = "gemma" in model_name.lower()
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    if has_flash or not is_gemma:
        if not has_flash and not attention_backend:
            os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
            os.environ.setdefault("VLLM_USE_V1", "0")
        return
    if Version(vllm.__version__).release < (0, 11, 0):
        sys.exit(
            f"ERROR: vLLM {vllm.__version__} cannot run Gemma 3 correctly on this GPU "
            "(XFormers disables its sliding-window attention, and FlexAttention only "
            "supports sliding window from 0.11.0). Install vllm==0.11.0 "
            "(torch 2.8.0, transformers<5) or run with --backend hf."
        )
    os.environ["VLLM_USE_V1"] = "1"
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION")


class HFRunner:
    """Sequential Hugging Face transformers generation, used where vLLM cannot run the model correctly."""

    def __init__(self, model_name, dtype, processor):
        import torch
        from transformers import AutoModelForImageTextToText

        torch_dtype = {
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }[dtype]
        self.torch = torch
        self.processor = processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            device_map="cuda",
        ).eval()

    def generate(self, requests, stops):
        results = []
        for inputs, max_tokens, temperature, *_ in requests:
            images = inputs["multi_modal_data"]["image"]
            encoded = self.processor(
                text=inputs["prompt"],
                images=images if images else None,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)
            if "pixel_values" in encoded:
                encoded["pixel_values"] = encoded["pixel_values"].to(self.model.dtype)
            prompt_len = encoded["input_ids"].shape[1]
            sample = temperature > 0
            with self.torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=sample,
                    temperature=temperature if sample else None,
                    top_p=0.95 if sample else None,
                    top_k=None,
                    repetition_penalty=1.05,
                    eos_token_id=stops,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            new_tokens = output[0, prompt_len:]
            text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            hit_stop = len(new_tokens) > 0 and int(new_tokens[-1]) in stops
            reason = "stop" if hit_stop or len(new_tokens) < max_tokens else "length"
            results.append((text, reason))
        return results


def load_model(args, max_pages):
    from transformers import AutoProcessor

    args.dtype = resolve_dtype(args.dtype, args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if args.backend == "hf":
        print(f"Using dtype: {args.dtype} | backend: hf")
        llm = HFRunner(args.model, args.dtype, processor)
    else:
        configure_vllm_env(args.model, args.attention_backend)
        use_flex = os.environ.get("VLLM_ATTENTION_BACKEND") == "FLEX_ATTENTION"
        if use_flex:
            os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
            os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")
            import tempfile
            _patch_dir = tempfile.mkdtemp(prefix="vllm_flex_patch_")
            with open(os.path.join(_patch_dir, "sitecustomize.py"), "w", encoding="utf-8") as _f:
                _f.write(
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
            _prev_pp = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = f"{_patch_dir}:{_prev_pp}" if _prev_pp else _patch_dir
            import torch
            import torch._dynamo.config as _dc
            _dc.cache_size_limit = 512
            _dc.accumulated_cache_size_limit = 4096
            if hasattr(_dc, "recompile_limit"):
                _dc.recompile_limit = 512
            import torch._inductor.config as _ic
            _ic.triton.cudagraphs = False
            _ic.triton.cudagraph_trees = False
            import torch.nn.attention.flex_attention as _tfa
            _orig_cbm = _tfa.create_block_mask

            def _eager_create_block_mask(*cbm_args, **cbm_kwargs):
                torch.compiler.cudagraph_mark_step_begin()
                cbm_kwargs["_compile"] = False
                return _orig_cbm(*cbm_args, **cbm_kwargs)

            _tfa.create_block_mask = _eager_create_block_mask
            import vllm.v1.attention.backends.flex_attention as vllm_flex
            vllm_flex.create_block_mask = _eager_create_block_mask
            if hasattr(vllm_flex, "create_block_mask_compiled"):
                vllm_flex.create_block_mask_compiled = _eager_create_block_mask
        enforce_eager = (
            args.enforce_eager if args.enforce_eager is not None else use_flex
        )
        print(
            f"Using dtype: {args.dtype} | backend: vllm | "
            f"V1={os.environ.get('VLLM_USE_V1', 'default')} | "
            f"attention={os.environ.get('VLLM_ATTENTION_BACKEND', 'default')} | "
            f"enforce_eager={enforce_eager}"
        )
        from vllm import LLM

        # Eager create_block_mask materializes a dense (step tokens x KV cache tokens)
        # mask, so FLEX_ATTENTION needs small prefill chunks and a small KV cache.
        max_batched = args.max_num_batched_tokens or (2048 if use_flex else None)
        gpu_util = args.gpu_util if args.gpu_util is not None else (0.70 if use_flex else 0.90)
        extra = {"max_num_batched_tokens": max_batched} if max_batched else {}
        print(f"gpu_util={gpu_util} | max_num_batched_tokens={max_batched or 'default'}")
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            gpu_memory_utilization=gpu_util,
            **extra,
            limit_mm_per_prompt={"image": max_pages},
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab = tokenizer.get_vocab()
    stops = []
    for token in ("<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"):
        if token in vocab:
            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid not in stops:
                stops.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stops:
        stops.append(tokenizer.eos_token_id)
    return llm, processor, stops


def make_sampling_params(max_tokens, temperature, stops, schema=None, backend=None):
    from vllm import SamplingParams

    common = {
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "repetition_penalty": 1.05,
        "max_tokens": max_tokens,
        "stop_token_ids": stops,
    }
    if schema is None:
        return SamplingParams(**common)
    try:
        from vllm.sampling_params import GuidedDecodingParams

        gd_kwargs = {"json": schema}
        if backend:
            gd_kwargs["backend"] = backend
        return SamplingParams(
            **common, guided_decoding=GuidedDecodingParams(**gd_kwargs)
        )
    except Exception:
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return SamplingParams(
                **common, structured_outputs=StructuredOutputsParams(json=schema)
            )
        except Exception:
            return SamplingParams(**common)


def generate_batch(llm, stops, requests, backend=None):
    if not requests:
        return []
    if isinstance(llm, HFRunner):
        return llm.generate(requests, stops)
    inputs = [req[0] for req in requests]
    has_schema = any(len(req) > 3 and req[3] is not None for req in requests)
    if has_schema:
        try:
            params = [
                make_sampling_params(m, t, stops, s, backend=backend)
                for _, m, t, s in requests
            ]
            outputs = llm.generate(inputs, params, use_tqdm=False)
            return [
                (o.outputs[0].text, o.outputs[0].finish_reason) if o.outputs else ("", None)
                for o in outputs
            ]
        except Exception as exc:
            exc_str = f"{type(exc).__name__}: {exc}".strip(": ")
            print(f"WARNING: guided JSON decoding failed ({exc_str}); running without schema")

    params = [make_sampling_params(m, t, stops, None) for req in requests for m, t in [(req[1], req[2])]]
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
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1204224,
        help="Per-image pixel cap before Gemma 3 vision encoding",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--text-token-reserve", type=int, default=1800)
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization (default: 0.70 with FLEX_ATTENTION, else 0.90)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Prefill chunk size (default: 2048 with FLEX_ATTENTION to bound block-mask memory)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("half", "bfloat16", "float32", "auto"),
        help="Model dtype. Default: float32 for Gemma on GPUs without bf16 (V100), bfloat16 on Ampere+",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=("vllm", "hf"),
        help="Inference engine. Gemma 3 on V100 with vLLM requires vllm>=0.11.0; 'hf' is a slow fallback",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Override VLLM_ATTENTION_BACKEND (default: FLEX_ATTENTION for Gemma on pre-Ampere GPUs)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable CUDA graphs (default: True when using FLEX_ATTENTION to avoid CUDAGraph tensor-overwrite errors)",
    )
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
        help="Require every question to combine at least 2 pages in source_path",
    )
    parser.add_argument(
        "--single-page-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify each question with single-page probes and reject if 1 page can answer it (default: True)",
    )
    parser.add_argument(
        "--raw-json-only",
        action="store_true",
        help="Write only {'result': [...]} per JSONL line without group metadata wrapper",
    )
    parser.add_argument(
        "--prefill-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefill assistant turn with '{' to force immediate JSON output (default: True)",
    )
    parser.add_argument(
        "--guided-decoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use vLLM guided JSON decoding (default: False, as xgrammar has known bugs with Gemma 3)",
    )
    parser.add_argument(
        "--guided-backend",
        default="outlines",
        choices=("outlines", "xgrammar", "lm-format-enforcer"),
        help="Guided decoding backend if --guided-decoding is enabled (default: outlines)",
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
        path_to_page = {
            sp: page for (page, _), sp in zip(job["window"], job["source_paths"])
        }
        enriched_items = None
        if items is not None:
            enriched_items = []
            for item in items:
                sp = item["source_path"]
                sp_list = sp if isinstance(sp, list) else [sp]
                pages = [path_to_page[p] for p in sp_list if p in path_to_page]
                enriched_items.append({**item, "source_pages": pages})
        payload = {
            "group_key": job["key"],
            "document": job["doc"],
            "pages": job["pages"],
            "source_paths": job["source_paths"],
            "attempts": attempt,
            "qa": {"result": enriched_items} if enriched_items is not None else None,
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
    needed = max_pages * TOKENS_PER_IMAGE + args.max_tokens + args.text_token_reserve
    if needed > args.max_model_len:
        sys.exit(
            f"ERROR: a {max_pages}-page group needs ~{needed} tokens but "
            f"--max-model-len is {args.max_model_len}. Raise --max-model-len or "
            "lower --max-tokens."
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
                        "schema": build_schema(
                            source_paths, args.require_multipage
                        ),
                        "locked_slots": {1: None, 2: None, 3: None},
                        "accepted": None,
                        "duplicates": [],
                        "feedback": "",
                        "raw": "",
                        "attempt": 0,
                    }
                )

            use_prefill = args.prefill_json and not args.guided_decoding
            for attempt in range(1, args.attempts + 1):
                active = [job for job in jobs if job["accepted"] is None]
                if not active:
                    break
                temp = 0.0 if attempt == 1 else min(0.6, 0.15 * attempt)

                requests = [
                    (
                        build_request(
                            processor,
                            job,
                            seen_by_doc[job["doc"]],
                            prefill_brace=use_prefill,
                        ),
                        args.max_tokens,
                        temp,
                        job["schema"] if args.guided_decoding else None,
                    )
                    for job in active
                ]
                outputs = generate_batch(llm, stops, requests, backend=args.guided_backend)
                if ok == 0 and all(
                    reason == "length" and len(set((text or "").strip())) <= 3
                    for text, reason in outputs
                ):
                    sys.exit(
                        "ERROR: the model returned only empty or repeated tokens until max_tokens. "
                        "Empty output = fp16 overflow (use --dtype float32 on V100). "
                        "Repetition = broken sliding-window attention (needs vllm>=0.11.0 "
                        "with FLEX_ATTENTION, or --backend hf)."
                    )
                slot_candidates = []
                parsed_jobs = []

                for job, (text, finish_reason) in zip(active, outputs):
                    job["attempt"] = attempt
                    job["raw"] = text
                    try:
                        data = parse_json(text, prefilled_brace=use_prefill)
                    except ValueError as exc:
                        suffix = (
                            " (output hit max_tokens limit; keep answers concise)"
                            if finish_reason == "length"
                            else ""
                        )
                        job["feedback"] = (
                            f"{exc}{suffix} | preview: {repr(text[:120]) if text else '<empty>'}"
                        )
                        continue

                    errors, new_cands = validate_candidates_per_slot(
                        data,
                        job,
                        seen_by_doc[job["doc"]],
                        args.dup_threshold,
                        args.require_multipage,
                    )
                    if "all" in errors:
                        job["feedback"] = errors["all"]
                        continue

                    parsed_jobs.append((job, errors, new_cands))
                    for idx, item in new_cands.items():
                        slot_candidates.append((job, idx, item))

                if args.single_page_check and slot_candidates:
                    leaks = probe_single_page_leaks(
                        llm,
                        processor,
                        stops,
                        slot_candidates,
                        guided_decoding=args.guided_decoding,
                        guided_backend=args.guided_backend,
                        prefill_brace=use_prefill,
                    )
                else:
                    leaks = {}

                for job, errors, new_cands in parsed_jobs:
                    for idx, item in new_cands.items():
                        if (job["key"], idx) in leaks:
                            errors[idx] = leaks[(job["key"], idx)]
                        else:
                            job["locked_slots"][idx] = item

                    if all(job["locked_slots"][i] is not None for i in (1, 2, 3)):
                        cleaned = [
                            strip_internal_keys(job["locked_slots"][i])
                            for i in (1, 2, 3)
                        ]
                        job["accepted"] = cleaned
                        seen_by_doc[job["doc"]].extend(
                            normalize_text(item["query"]) for item in cleaned
                        )
                        write_record(
                            stream,
                            job,
                            attempt,
                            cleaned,
                            [],
                            None,
                            args.raw_json_only,
                        )
                        ok += 1
                        cats = ", ".join(item["category"] for item in cleaned)
                        print(f"OK   {job['key']} (attempt {attempt}): {cats}")
                    else:
                        job["feedback"] = " | ".join(
                            errors[i] for i in sorted(errors)
                        )

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
