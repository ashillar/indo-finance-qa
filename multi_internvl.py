"""Generate multi-page QA pairs from document page images with InternVL and vLLM.

Usage:
    python multi_internvl.py --folder bi_multi --groups-file bi_multi_groups.txt --out bi_multipage_internvl.jsonl
    python multi_internvl.py --folder ojk_multi --groups-file ojk_multi_groups.txt --out ojk_multipage_internvl.jsonl
    python multi_internvl.py --folder bi_multi --dry-run
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

DEFAULT_MODEL = "OpenGVLab/InternVL2_5-4B"
DEFAULT_OUT = "bi_multipage_internvl.jsonl"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)
# InternVL dynamic resolution: each 448x448 tile (plus thumbnail) = 256 vision tokens.
TOKENS_PER_TILE = 256

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

# Reject explicit references to table/figure/page numbers in queries or answers.
FIG_TABLE_NUM_RE = re.compile(
    r"\b(?:tabel|grafik|gambar|bagan|figure|table|chart)\s+"
    r"(?:no\.?\s*)?(?:\d+|[ivxlcdm]+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
PAGE_REF_RE = re.compile(
    r"\b(?:halaman|hal\.?|page)\s*(?:\d+|[ab]|[ivxlcdm]+)\b",
    re.IGNORECASE,
)
# Reject template/placeholder answers copied from instructions.
PLACEHOLDER_ANS_RE = re.compile(
    r"(?:\b(?:jawaban|fakta/angka|angka\s+spesifik)\s+(?:yang|dengan|dari)\b"
    r"|\b(?:halaman\s+[ab]|page\s+[ab]|indikator\s+[ab]|komponen\s+[ab])\b"
    r"|\bpersisannya\b|\.\.\.)",
    re.IGNORECASE,
)
# Reject yes/no confirmation questions ("Apakah ... mencapai X%?") which spoil the answer.
YES_NO_QUERY_RE = re.compile(r"\bapakah\b", re.IGNORECASE)
# Require calculation questions to actually ask for a computation.
CALC_QUERY_WORD_RE = re.compile(
    r"\b(?:selisih|total|jumlah|rata-rata|rasio|perbandingan|proporsi|pangsa|"
    r"persentase\s+perubahan|beda)\b",
    re.IGNORECASE,
)
# Detect specific numeric values leaked inside a calculation query (excluding 4-digit years).
LEAKED_VALUE_RE = re.compile(
    r"(?:rp\s*\d[\d.,]*|\d+[.,]\d+\s*%?|\b\d+\s*%)",
    re.IGNORECASE,
)

CATEGORY_EXAMPLES = """
Contoh gaya yang BENAR (dari dokumen fiktif "Laporan Perekonomian Kalimantan Barat Triwulan III 2019" — HANYA contoh format, JANGAN menyalin angka/topik ini):
- [Text-Text Inference]
  evidence_page_1: "pangsa konsumsi rumah tangga mencapai 54,8% terhadap PDRB (hanya di narasi halaman pertama)"
  evidence_page_2: "indeks keyakinan konsumen meningkat menjadi 124,6 (hanya di halaman kedua)"
  query: "Pada Laporan Perekonomian Kalimantan Barat Triwulan III 2019, bagaimana kaitan antara pangsa konsumsi rumah tangga terhadap PDRB dan perkembangan indeks keyakinan konsumen?"
  answer: "Konsumsi rumah tangga memiliki pangsa sebesar 54,8% terhadap PDRB, yang didukung oleh peningkatan indeks keyakinan konsumen menjadi 124,6." (long)
- [Table-Numerical Calculation]
  evidence_page_1: "pangsa kredit modal kerja tercatat 46,2% terhadap total kredit (hanya di halaman pertama)"
  evidence_page_2: "pangsa kredit investasi tercatat 28,9% terhadap total kredit (hanya di halaman kedua)"
  query: "Dalam Laporan Perekonomian Kalimantan Barat Triwulan III 2019, berapakah selisih antara pangsa kredit modal kerja dan pangsa kredit investasi terhadap total kredit?"
  answer: "Pangsa kredit modal kerja adalah 46,2% dan pangsa kredit investasi adalah 28,9%, sehingga selisihnya sebesar 46,2% - 28,9% = 17,3 poin persentase." (short)
"""

PROMPT_TEMPLATE = """Document: {doc_name}
Pages in this group:
{source_list}

Generate THREE multi-page QA pairs (`q1`, `q2`, `q3`) in Bahasa Indonesia based STRICTLY on the provided page images.

CRITICAL RULES:
1. STRICT PAGE-UNIQUE EVIDENCE (NO SINGLE-PAGE LEAKS, NO HALLUCINATION):
   - Use ONLY facts, entities, and exact numbers visible in the current page images above.
   - `evidence_page_1` MUST be a short exact quote/fact (max 20 words) that appears ONLY on the FIRST assigned page (`Page A`) and does NOT appear anywhere on `Page B` (not even in a summary table on `Page B`).
   - `evidence_page_2` MUST be a DIFFERENT short exact quote/fact (max 20 words) that appears ONLY on the SECOND assigned page (`Page B`) and does NOT appear anywhere on `Page A`.
   - IMPORTANT: If `Page B` contains a table that repeats headline numbers from `Page A`'s narrative, do NOT use those repeated headline numbers for `evidence_page_1`. Instead, pick a detail, share (`pangsa`), chart value, or narrative explanation that is UNIQUE to `Page A`.
   - Never copy facts from the wrong page, and never invent numbers.
2. GENUINE MULTI-PAGE QUESTIONS:
   - Each question (`query`) MUST explicitly ask about BOTH the indicator/topic from `evidence_page_1` (on `Page A`) AND the indicator/topic from `evidence_page_2` (on `Page B`), so that the question CANNOT be answered from either page alone.
   - Both `evidence_page_1` and `evidence_page_2` MUST be explicitly stated with their exact indicator names and numbers/facts in `answer`.
3. QUESTION QUALITY (NO YES/NO, NO LEAKED ANSWERS, NO PAGE/TABLE NUMBERS):
   - NEVER use "Apakah" or write yes/no confirmation questions. Use "Bagaimana", "Berapa", "Berapakah", "Mengapa", or "Komponen/daerah mana".
   - Do NOT leak the answer's numbers inside the `query` (e.g., do NOT ask "Apakah X mencapai 27,37%?"). Ask for the value/comparison/calculation and put the numbers in `answer`.
   - For `Table-Numerical Calculation` and `Chart-Numerical Calculation`, the `query` MUST ask to compute a difference (`selisih`), sum (`total`/`jumlah`), average (`rata-rata`), or ratio (`rasio`) between a metric unique to Page A and a metric unique to Page B, without stating the two numbers in the `query`. The `answer` MUST state both source numbers and the accurate computed result.
   - Do NOT mention page numbers ("halaman 70", "halaman A") or table/figure/chart numbers ("Tabel 1", "Grafik 2.1") in `query` or `answer`. Name the indicator, region, and period instead.
   - NEVER write placeholder/template text in `answer` (such as "Jawaban yang menyebutkan...").
4. DISTINCT TOPICS & ACCURATE CATEGORIES:
   - `q1`, `q2`, and `q3` MUST cover THREE COMPLETELY DIFFERENT indicators, table rows, or chart series, using different numbers across `q1`, `q2`, and `q3`.
   - Pick `Chart-*` ONLY if the pages actually contain a visual chart/graph; pick `Table-*` ONLY if the pages contain a data table.
   - `source_path` MUST be the exact 2-element array of file paths for the slot's assigned page pair.

Assigned Slots:
{slot_instructions}

Categories:
- Text-Text Inference: Synthesize narrative explanations, causes, or policy evaluations across both pages.
- Chart-Information Extraction: Extract and compare values from charts/graphs across both pages.
- Chart-Numerical Calculation: Compute selisih, total, rata-rata, or rasio from chart values across both pages.
- Chart-Time-Sensitive: Compare multi-period chart trends across both pages.
- Table-Numerical Calculation: Compute selisih, total, rata-rata, or rasio from table numbers across both pages.
- Table-Comparison and Sorting: Rank, sort, or compare highest/lowest entities using table values across both pages.
{category_examples}{feedback_block}
Output valid JSON only (no Markdown fences):
{{"q1": {{"evidence_page_1": "...", "evidence_page_2": "...", "query": "...", "category": "Text-Text Inference", "answer": "...", "answer_type": "long", "source_path": ["...", "..."]}}, "q2": {{"evidence_page_1": "...", "evidence_page_2": "...", "query": "...", "category": "Table-Numerical Calculation", "answer": "...", "answer_type": "short", "source_path": ["...", "..."]}}, "q3": {{"evidence_page_1": "...", "evidence_page_2": "...", "query": "...", "category": "Table-Comparison and Sorting", "answer": "...", "answer_type": "short", "source_path": ["...", "..."]}}}}"""

PROBE_PROMPT = """Gambar di atas adalah SATU halaman dokumen saja.
Periksa secara ketat HANYA berdasarkan teks, tabel, dan grafik yang terlihat di halaman ini:
1. Fakta 1: "{ev1}" -> Apakah fakta dan angka spesifik pada Fakta 1 ini tertulis di halaman ini? (ev1_on_page: true atau false)
2. Fakta 2: "{ev2}" -> Apakah fakta dan angka spesifik pada Fakta 2 ini tertulis di halaman ini? (ev2_on_page: true atau false)
3. Pertanyaan: "{query}" -> Dapatkah SELURUH bagian dari pertanyaan ini dijawab secara LENGKAP hanya dengan melihat halaman ini saja tanpa halaman lain? Jika halaman ini hanya memuat sebagian informasi (misalnya hanya salah satu indikator), isi "answerable" dengan false dan "answer" dengan "".

Keluarkan HANYA JSON valid:
{{"ev1_on_page": true atau false, "ev2_on_page": true atau false, "answerable": true atau false, "answer": "..."}}"""

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
# Prompt & Schema Construction (InternVL Format)
# --------------------------------------------------------------------------


def format_internvl_chat(tokenizer, user_text, prefill_brace=False):
    """Apply InternVL chat template around a user message containing <image> placeholders."""
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
                "evidence_page_1": {"type": "string"},
                "evidence_page_2": {"type": "string"},
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


def build_request(tokenizer, job, avoid_queries, prefill_brace=False):
    source_paths = job["source_paths"]
    page_to_path = {
        page: sp for (page, _), sp in zip(job["window"], source_paths)
    }
    source_list = "\n".join(
        f"- Image-{idx} [Page {page}]: `{sp}`"
        for idx, ((page, _), sp) in enumerate(zip(job["window"], source_paths), 1)
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

    # InternVL multi-image format: one <image> placeholder per image in the text prompt.
    image_headers = [
        f"[START OF IMAGE-{idx} | Page {page} | source_path: {sp}]\n<image>\n[END OF IMAGE-{idx} | Page {page}]"
        for idx, ((page, _), sp) in enumerate(zip(job["window"], source_paths), 1)
    ]
    user_text = "\n".join(image_headers) + "\n\n" + prompt_text
    prompt = format_internvl_chat(tokenizer, user_text, prefill_brace=prefill_brace)

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
    page_match = re.search(r"(?:^|[_\s-])p?0*(\d+)$", cand_stem, re.IGNORECASE)
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
    return {k: item[k] for k in ITEM_KEYS if k in item}


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
    if FIG_TABLE_NUM_RE.search(query) or FIG_TABLE_NUM_RE.search(answer):
        return (
            f"q{idx} explicitly mentions a figure or table number; "
            "describe the indicator/content instead"
        ), None
    if YES_NO_QUERY_RE.search(query):
        return (
            f"q{idx} uses a yes/no question ('Apakah...'); "
            "rewrite using 'Bagaimana', 'Berapa', 'Mengapa', or 'Komponen/daerah mana' "
            "without stating the answer's numbers inside the question"
        ), None

    if require_multipage and len(valid_paths) >= 2:
        if len(ev1) < 5 or len(ev2) < 5:
            return (
                f"q{idx} must provide non-empty 'evidence_page_1' (unique to Page {pa}) "
                f"and 'evidence_page_2' (unique to Page {pb})"
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
        if not evidence_supported_in_text(ev1, answer, context_text=combined_qa) or not evidence_supported_in_text(ev2, answer, context_text=combined_qa):
            return (
                f"q{idx} answer does not accurately include both pages' evidences; the answer MUST "
                f"explicitly state the exact indicator names and numbers from BOTH 'evidence_page_1' "
                f"(Page {pa}) and 'evidence_page_2' (Page {pb})"
            ), None
        if not query_combines_evidences(query, ev1, ev2):
            return (
                f"q{idx} query only asks about one indicator/page; rewrite q{idx} 'query' so the question "
                f"explicitly asks about BOTH the topic in 'evidence_page_1' (Page {pa}) AND "
                f"the topic in 'evidence_page_2' (Page {pb})"
            ), None
        slot_nums = frozenset(ev1_nums | ev2_nums)
        if slot_nums and any(
             prev_nums and (slot_nums == prev_nums or len(slot_nums & prev_nums) >= 2)
            for prev_nums in used_num_sets
        ):
            return (
                f"q{idx} reuses the same evidence numbers as another question in this group; "
                f"pick a completely different metric/row on Page {pa} and Page {pb}"
            ), None
    else:
        slot_nums = frozenset()

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
        if len(expected_pair_paths) >= 2:
            norm_source = expected_pair_paths
        elif not isinstance(norm_source, list) or len(norm_source) < 2:
            if len(valid_paths) == 2:
                norm_source = list(valid_paths)
            else:
                return (
                    f"q{idx} only cites 1 page ({norm_source!r}); "
                    "every question must combine 2 pages in source_path"
                ), None

    doc_name = job.get("doc", "")
    if is_near_duplicate(query, batch_queries, dup_threshold, doc_name=doc_name):
        return (
            f"q{idx} repeats another question in this group ({query!r}); "
            f"rewrite q{idx} to ask about a completely different indicator or metric "
            f"combining Page {pa} and Page {pb}"
        ), None
    if is_near_duplicate(query, seen_queries, dup_threshold, doc_name=doc_name):
        return f"q{idx} repeats a question from an earlier group", None

    used_categories.add(category)
    batch_queries.append(query)
    if slot_nums:
        used_num_sets.append(slot_nums)
    answer_type = normalize_answer_type(raw_type, answer)

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
        # If both evidences have distinct content words, the query should touch both sides
        # (or share their common topic word plus ask a multi-part/comparison question).
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
    tokenizer,
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
            user_text = f"<image>\n{probe_text}"
            prompt = format_internvl_chat(
                tokenizer, user_text, prefill_brace=prefill_brace
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
                other_page = pb if page == pa else pa
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

        # Also verify that evidence_page_1 is actually on Page pa and evidence_page_2 is on Page pb.
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


def _patch_pil_fromarray():
    import numpy as np
    from PIL import Image

    orig_fromarray = Image.fromarray
    if getattr(orig_fromarray, "_int64_patched", False):
        return

    def _safe_fromarray(obj, mode=None):
        if isinstance(obj, np.ndarray) and obj.dtype != np.uint8:
            if obj.ndim == 3 and obj.shape[-1] in (1, 3, 4) and np.issubdtype(obj.dtype, np.integer):
                obj = np.clip(obj, 0, 255).astype(np.uint8)
        return orig_fromarray(obj, mode=mode)

    _safe_fromarray._int64_patched = True
    Image.fromarray = _safe_fromarray


def load_model(args, max_pages):
    _patch_pil_fromarray()
    from transformers import AutoTokenizer
    from vllm import LLM

    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "gpu_memory_utilization": args.gpu_util,
        "limit_mm_per_prompt": {"image": max_pages, "video": 0},
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": True,
    }
    if args.max_dynamic_patch is not None:
        llm_kwargs["mm_processor_kwargs"] = {
            "max_dynamic_patch": args.max_dynamic_patch
        }

    try:
        llm = LLM(**llm_kwargs)
    except TypeError:
        llm_kwargs.pop("mm_processor_kwargs", None)
        llm = LLM(**llm_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    stops = []
    for token in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if token in vocab:
            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid not in stops:
                stops.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stops:
        stops.append(tokenizer.eos_token_id)
    return llm, tokenizer, stops


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
            print(f"WARNING: guided JSON decoding unavailable ({exc_str}); running without schema")
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
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1204224,
        help="Per-image pixel cap before InternVL dynamic tiling",
    )
    parser.add_argument(
        "--max-dynamic-patch",
        type=int,
        default=6,
        help="Max 448x448 tiles per image for InternVL (each tile + thumbnail = 256 tokens; default: 6)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Context length. 32768 lets a batch of 3-page groups run together on a 32GB V100",
    )
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--text-token-reserve", type=int, default=1800)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--dtype", default="half", choices=("half", "bfloat16", "float32", "auto"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable CUDA graphs in vLLM (default: False)",
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
        default=False,
        help="Prefill assistant turn with '{' when guided decoding is disabled (default: False)",
    )
    parser.add_argument(
        "--guided-decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM guided JSON decoding (default: True)",
    )
    parser.add_argument(
        "--guided-backend",
        default=None,
        choices=("outlines", "xgrammar", "lm-format-enforcer"),
        help="Guided decoding backend if --guided-decoding is enabled",
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
    max_tiles = (args.max_dynamic_patch or 6) + 1
    tokens_per_image = max_tiles * TOKENS_PER_TILE
    needed = max_pages * tokens_per_image + args.max_tokens + args.text_token_reserve
    if needed > args.max_model_len:
        sys.exit(
            f"ERROR: a {max_pages}-page group with --max-dynamic-patch {args.max_dynamic_patch} "
            f"needs up to ~{needed} tokens but --max-model-len is {args.max_model_len}. "
            "Raise --max-model-len or lower --max-dynamic-patch / --max-tokens."
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

    llm, tokenizer, stops = load_model(args, max_pages)

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
                            tokenizer,
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
                        tokenizer,
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
