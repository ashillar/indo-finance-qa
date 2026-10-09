"""Generate multi-page QA pairs from page images with Qwen2.5-VL and vLLM.

Every question is meant to need information from at least two pages. The
pipeline enforces this with four layers of checks:

1. Schema: each question carries evidence (page, kind, quote) that the model
   fills BEFORE it writes the answer. Each question is given two different
   pages of its group, and the schema requires one evidence field named
   after each page ("halaman_55", "halaman_56"), so every question cites
   two pages. "source_pages" is filled from the evidence.
2. Validation: cited pages must equal the evidence pages, every cited page must
   contribute something the other pages do not, numbers in the answer must come
   from the evidence, and the category must match the kind of evidence.
3. OCR (optional, --ocr): evidence numbers must appear on the page they cite,
   and no single page may contain every number a question relies on.
4. Single-page probe (optional, --single-page-check): each question is asked
   again with one cited page at a time. If one page alone gives the same
   answer, the question is not multi-page and is rejected.

Rejected windows are retried, with the rejection reason added to the prompt.
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

from PIL import Image

# V100 / vLLM
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_V1", "0")

# vLLM and transformers are imported lazily so --dry-run works on machines
# without a GPU or vLLM installed.


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SLOTS = ("q1", "q2", "q3")
FIELDS = (
    "query",
    "category",
    "evidence",
    "answer",
    "answer_type",
    "source_pages",
)
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)
# Qwen2.5-VL: 14px patch, spatial merge 2 -> 28x28 pixels per vision token.
QWEN_PIXELS_PER_TOKEN = 28 * 28

KINDS = ("Teks", "Grafik", "Tabel")

# Category -> (evidence kind it requires, description shown to the model).
CATEGORIES = {
    "Text-Text Inference": (
        "Teks",
        "Ekstraksi atau penalaran berdasarkan informasi teks.",
    ),
    "Chart-Information Extraction": (
        "Grafik",
        "Mengambil metrik atau ciri utama dari grafik.",
    ),
    "Chart-Numerical Calculation": (
        "Grafik",
        "Perhitungan dari angka grafik, misalnya laju pertumbuhan, suku "
        "bunga, atau total biaya.",
    ),
    "Chart-Time-Sensitive": (
        "Grafik",
        "Deskripsi tren, hubungan sebab-akibat, urutan peristiwa, "
        "frekuensi, atau durasi dari grafik.",
    ),
    "Table-Numerical Calculation": (
        "Tabel",
        "Perhitungan dari data tabel, misalnya laju pertumbuhan, suku "
        "bunga, atau total biaya.",
    ),
    "Table-Comparison and Sorting": (
        "Tabel",
        "Membandingkan atau mengurutkan entitas berdasarkan kriteria "
        "tertentu (misalnya tingkat imbal hasil atau risiko).",
    ),
}
CATEGORY_NAMES = list(CATEGORIES)
CALCULATION_CATEGORIES = {
    "Chart-Numerical Calculation",
    "Table-Numerical Calculation",
}
CATEGORY_BLOCK = "\n".join(
    f"- {name}: {description} (bukti wajib berjenis \"{kind}\")"
    for name, (kind, description) in CATEGORIES.items()
)

MULTI_PAGE_RULE = """{who} WAJIB memakai informasi dari minimal DUA halaman
berbeda dan TIDAK BOLEH bisa dijawab hanya dari satu halaman.{others}
- Setiap halaman yang dikutip harus menyumbang informasi yang BERBEDA. Angka
  yang sama yang muncul di dua halaman TIDAK dihitung sebagai lintas halaman.
- Pola yang baik: membandingkan angka dari satu halaman dengan angka dari
  halaman lain; menghitung selisih atau rasio antara angka di dua halaman;
  mengaitkan penjelasan teks di satu halaman dengan data tabel atau grafik
  di halaman lain.
- Pertanyaan harus menyebut kedua informasi tersebut, sehingga jelas bahwa
  keduanya dibutuhkan untuk menjawab."""

SINGLE_PAGE_RULE = (
    "Hanya tersedia satu halaman, jadi semua pertanyaan berasal dari halaman "
    "tersebut. Ambil bagian atau aspek yang berbeda-beda."
)

INSTRUCTIONS = """
Anda menerima {n_pages} gambar halaman dari SATU dokumen yang sama. Setiap
gambar diberi label [Halaman N]. Nomor halaman yang sah hanya: {page_list}.
Gunakan nomor label tersebut, BUKAN nomor halaman yang tercetak di gambar.

TUGAS: buat TEPAT TIGA pasangan pertanyaan dan jawaban dalam Bahasa Indonesia,
diisi pada slot q1, q2, dan q3.

SYARAT LINTAS HALAMAN (paling penting):
{multi_page_rule}

ATURAN LAIN:
1. Setiap pertanyaan memakai kategori yang BERBEDA.
2. Tanyakan fakta atau informasi nyata dari isi dokumen, bukan tentang elemen
   visual atau struktur dokumen.
3. Jangan gunakan frasa "tabel ini", "tabel di atas", "grafik ini",
   "gambar ini", "teks ini", "teks di atas", "dokumen ini", "halaman ini",
   "berdasarkan tabel", "berdasarkan grafik", "menurut data di atas",
   nomor tabel, atau nomor grafik/gambar.
4. Sebutkan wilayah, periode, sektor, indikator, institusi, atau entitas yang
   dibahas.
5. Jangan mengarang angka atau fakta. Salin angka persis seperti yang
   terlihat dan pastikan kolom periodenya benar.
6. Perhitungan hanya memakai angka yang terlihat; tulis langkah singkatnya
   di jawaban.
7. "answer_type" bernilai "pendek" untuk jawaban singkat, atau "panjang" bila
   perlu penjelasan atau beberapa informasi.
8. Jangan membuat pertanyaan yang isinya hampir sama.

URUTAN PENGISIAN SETIAP PERTANYAAN:
a. "evidence": satu bukti dari SETIAP halaman yang ditentukan untuk
   pertanyaan tersebut (lihat HALAMAN PER PERTANYAAN), dengan kunci
   "halaman_<nomor>". Bukti dari kedua halaman harus berisi informasi yang
   BERBEDA. Setiap bukti berisi "kind" ("Teks", "Grafik", atau "Tabel") dan
   "quote" (kutipan singkat yang persis dari halaman itu, termasuk angka
   yang dipakai beserta nama baris/kolom atau periodenya). "bukti_lain"
   berisi bukti tambahan (dengan "page") bila perlu, atau [].
b. "answer": jawab HANYA berdasarkan evidence, dengan memakai informasi dari
   kedua halaman.

HALAMAN PER PERTANYAAN:
{pair_block}

KATEGORI:
{category_block}

{avoid_block}
{feedback_block}
Keluarkan HANYA JSON valid, tanpa Markdown atau ```json, dengan bentuk:
{{
  "result": {{
    "q1": {{
      "query": "<pertanyaan>",
      "category": "<salah satu kategori>",
      "evidence": {{
{evidence_example}
        "bukti_lain": []
      }},
      "answer": "<jawaban>",
      "answer_type": "<pendek|panjang>"
    }},
    "q2": {{ ...format yang sama, dengan halaman untuk q2... }},
    "q3": {{ ...format yang sama, dengan halaman untuk q3... }}
  }}
}}
""".strip()

PROBE_INSTRUCTIONS = """
Gambar di atas adalah SATU halaman dokumen.
Jawab pertanyaan berikut HANYA dengan informasi yang terlihat di halaman ini.
Jika halaman ini tidak memuat SEMUA informasi yang dibutuhkan, isi
"answerable" dengan false dan "answer" dengan "".

Pertanyaan: {query}

Keluarkan HANYA JSON valid: {{"answerable": true atau false, "answer": "..."}}
""".strip()

PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["answerable", "answer"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Numbers and text helpers
# --------------------------------------------------------------------------

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Period labels such as "IV-12", "III-2012" or "Tw.IV-2012" are not data.
PERIOD_RE = re.compile(r"\b(?:I{1,3}|IV)\s*-\s*\d{2,4}\b", re.IGNORECASE)
PAGE_REF_RE = re.compile(r"\bhalaman\s+\d+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def number_map(text):
    """Map comparable digit keys to the numbers written in the text.

    "6,75", "6.75" and "(6.75)" all become the key "675", so numbers match
    across Indonesian and English decimal styles. Single digits, years and
    period labels are ignored because they are not data values.
    """
    text = PAGE_REF_RE.sub(" ", PERIOD_RE.sub(" ", text or ""))
    found = {}
    for token in NUMBER_RE.findall(text):
        if YEAR_RE.fullmatch(token):
            continue
        digits = re.sub(r"\D", "", token).lstrip("0") or "0"
        if len(re.sub(r"\D", "", token)) < 2:
            continue
        found.setdefault(digits, token)
    return found


def number_keys(text):
    return set(number_map(text))


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# --------------------------------------------------------------------------
# Files and page windows
# --------------------------------------------------------------------------


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
    """Split pages into runs of nearby pages so unrelated sections stay apart."""
    if not pages:
        return []
    runs = [[pages[0]]]
    for item in pages[1:]:
        if item[0] - runs[-1][-1][0] <= max_gap:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def make_windows(pages, group_size, overlap):
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(pages) <= group_size:
        return [pages]

    overlap = max(0, min(overlap, group_size - 1))
    step = group_size - overlap
    windows = []
    for start in range(0, len(pages), step):
        window = pages[start : start + group_size]
        if len(window) < 2 and windows:
            # Never leave a one-page tail: a multi-page window needs 2 pages.
            window = pages[-2:]
        windows.append(window)
        if start + group_size >= len(pages):
            break
    return windows


def vision_tokens_per_image(max_pixels):
    return max(1, max_pixels // QWEN_PIXELS_PER_TOKEN)


def compute_safe_group_size(
    requested, max_model_len, max_pixels, max_tokens, text_reserve
):
    """Return a page count whose worst-case vision tokens fit the context."""
    if requested <= 0:
        raise ValueError("--group-size must be positive")
    if max_pixels < QWEN_PIXELS_PER_TOKEN:
        raise ValueError("--max-pixels is too small for one vision token")
    if max_tokens <= 0 or max_model_len <= 0:
        raise ValueError("token limits must be positive")

    tokens_per_image = vision_tokens_per_image(max_pixels)
    available = max_model_len - max_tokens - text_reserve
    if available < tokens_per_image:
        required = tokens_per_image + max_tokens + text_reserve
        raise ValueError(
            f"--max-model-len {max_model_len} cannot fit even one image "
            f"({tokens_per_image} vision + {max_tokens} output + "
            f"{text_reserve} text-reserve tokens). Use at least {required}, "
            "or reduce --max-pixels/--max-tokens."
        )
    return min(requested, available // tokens_per_image), tokens_per_image


def group_key(doc, pages):
    return f"{doc}::" + "-".join(f"p{page:03d}" for page in pages)


def load_image(path, max_edge, max_pixels=None):
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = max_edge / max(width, height)
    # Some vLLM versions drop mm_processor_kwargs (min/max_pixels), so cap
    # the pixel count here to keep the vision-token budget honest.
    if max_pixels and width * height * min(scale, 1) ** 2 > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
    if scale < 1:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.LANCZOS,
        )
    return image


# --------------------------------------------------------------------------
# Prompts and schemas
# --------------------------------------------------------------------------


def load_processor_and_stops(model):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=True
        )
        tokenizer = getattr(processor, "tokenizer", processor)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True
        )
        processor = tokenizer

    stops = []
    for token in ("<|endoftext|>", "<|im_end|>", "<|end|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if (
                token_id is not None
                and token_id >= 0
                and token_id != getattr(tokenizer, "unk_token_id", None)
            ):
                stops.append(token_id)
        except Exception:
            pass
    return processor, list(dict.fromkeys(stops))


def make_llm(args, group_size):
    mm_processor_kwargs = {}
    if args.min_pixels:
        mm_processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels:
        mm_processor_kwargs["max_pixels"] = args.max_pixels

    from vllm import LLM

    return LLM(
        model=args.model,
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        limit_mm_per_prompt={"image": group_size},
        trust_remote_code=True,
        tokenizer_mode="auto",
        enforce_eager=False,
        mm_processor_kwargs=mm_processor_kwargs or None,
    )


def chat_prompt(processor, page_numbers, instructions):
    content = []
    for page in page_numbers:
        content.append({"type": "image"})
        content.append({"type": "text", "text": f"[Halaman {page}]"})
    content.append({"type": "text", "text": instructions})
    try:
        return processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        parts = "".join(
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"[Halaman {page}]\n"
            for page in page_numbers
        )
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{parts}\n{instructions}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def build_group_prompt(
    processor,
    page_numbers,
    avoid_queries,
    min_multi,
    feedback=None,
    categories=None,
):
    if len(page_numbers) > 1:
        if min_multi >= 3:
            who, others = "SETIAP pertanyaan (q1, q2, dan q3)", ""
        else:
            who = f"Minimal {min_multi} dari 3 pertanyaan"
            others = " Pertanyaan lainnya boleh dari satu halaman."
        multi_page_rule = MULTI_PAGE_RULE.format(who=who, others=others)
    else:
        multi_page_rule = SINGLE_PAGE_RULE

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
            "PERCOBAAN SEBELUMNYA DITOLAK karena: "
            f"{feedback}\nPerbaiki kesalahan tersebut.\n"
        )

    pairs = slot_pairs(page_numbers)
    forced = categories or [None] * len(SLOTS)
    pair_block = "\n".join(
        f"- {slot}: gabungkan informasi dari "
        + " DAN ".join(f"halaman {page}" for page in pair)
        + " (isi "
        + " dan ".join(f'"halaman_{page}"' for page in pair)
        + ")"
        + (
            f". Kategori WAJIB: {category}, dengan minimal satu bukti "
            f'berjenis "{CATEGORIES[category][0]}"'
            if category
            else ""
        )
        for slot, pair, category in zip(SLOTS, pairs, forced)
    )
    evidence_example = "\n".join(
        f'        "halaman_{page}": {{"kind": "<Teks|Grafik|Tabel>", '
        f'"quote": "<kutipan dari halaman {page}>"}},'
        for page in pairs[0]
    )

    instructions = INSTRUCTIONS.format(
        n_pages=len(page_numbers),
        page_list=", ".join(map(str, page_numbers)),
        multi_page_rule=multi_page_rule,
        pair_block=pair_block,
        evidence_example=evidence_example,
        category_block=CATEGORY_BLOCK,
        avoid_block=avoid_block,
        feedback_block=feedback_block,
    )
    return chat_prompt(processor, page_numbers, instructions)


def slot_pairs(page_numbers):
    """Give q1, q2 and q3 each two different pages of the group.

    [38, 39] -> (38, 39) for every slot.
    [38, 39, 40] -> (38, 39), (39, 40), (38, 40).
    """
    pages = list(page_numbers)
    if len(pages) < 2:
        return [tuple(pages)] * len(SLOTS)
    adjacent = list(zip(pages, pages[1:]))
    others = [
        (a, b)
        for i, a in enumerate(pages)
        for b in pages[i + 2 :]
    ]
    pairs = adjacent + others
    return [pairs[i % len(pairs)] for i in range(len(SLOTS))]


def build_probe_prompt(processor, page, query):
    return chat_prompt(
        processor, [page], PROBE_INSTRUCTIONS.format(query=query)
    )


def item_schema(allowed_pages, pair, category=None):
    kind = {"type": "string", "enum": list(KINDS)}
    page_entry = {
        "type": "object",
        "properties": {"kind": kind, "quote": {"type": "string"}},
        "required": ["kind", "quote"],
        "additionalProperties": False,
    }
    extra_entry = {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "enum": allowed_pages},
            "kind": kind,
            "quote": {"type": "string"},
        },
        "required": ["page", "kind", "quote"],
        "additionalProperties": False,
    }
    # Property order matters: evidence is generated before the answer.
    # Do not add minItems/uniqueItems: xgrammar cannot compile them. One
    # required key per assigned page forces evidence from both pages.
    page_keys = [f"halaman_{page}" for page in pair]
    # A forced category is not generated at all (an enum would be stripped
    # for xgrammar); apply_forced_categories fills it in after parsing.
    category_property = (
        {"category": {"type": "string", "enum": CATEGORY_NAMES}}
        if category is None
        else {}
    )
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            **category_property,
            "evidence": {
                "type": "object",
                "properties": {
                    **{key: page_entry for key in page_keys},
                    "bukti_lain": {"type": "array", "items": extra_entry},
                },
                "required": page_keys + ["bukti_lain"],
                "additionalProperties": False,
            },
            "answer": {"type": "string"},
            "answer_type": {"type": "string", "enum": ["pendek", "panjang"]},
        },
        # source_pages is filled from the evidence in normalize_qa.
        "required": [
            field
            for field in FIELDS
            if field != "source_pages"
            and (field != "category" or category is None)
        ],
        "additionalProperties": False,
    }


def build_schema(allowed_pages, categories=None):
    if categories is None:
        categories = [None] * len(SLOTS)
    pairs = slot_pairs(allowed_pages)
    return {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    slot: item_schema(allowed_pages, pair, category)
                    for slot, pair, category in zip(SLOTS, pairs, categories)
                },
                "required": list(SLOTS),
                "additionalProperties": False,
            }
        },
        "required": ["result"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def extract_json(text):
    if not text or not text.strip():
        raise ValueError("Model returned empty output")

    text = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(
        r'(["\]\}])\s*\n\s*("(?:query|category|evidence|answer|answer_type|'
        r'source_pages|halaman_\d+|bukti_lain|page|kind|quote|'
        r'answerable)"\s*:)',
        r"\1,\n\2",
        text,
    )

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
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if quoted and char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

    partial = text[start:].rstrip()
    if partial.count('"') % 2:
        partial += '"'
    partial = re.sub(r",\s*$", "", partial)

    opens = []
    quoted = False
    escaped = False
    for char in partial:
        if escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char in "{[":
            opens.append(char)
        elif not quoted and char == "}" and opens and opens[-1] == "{":
            opens.pop()
        elif not quoted and char == "]" and opens and opens[-1] == "[":
            opens.pop()
    for char in reversed(opens):
        partial += "}" if char == "{" else "]"

    try:
        return json.loads(partial)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not repair truncated JSON: {exc}") from exc


def normalize_qa(data):
    if not isinstance(data, dict):
        raise ValueError("Root must be an object")
    result = data.get("result", data)
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        items = [result[slot] for slot in SLOTS if slot in result]
    else:
        raise ValueError("'result' must contain q1/q2/q3")
    for item in items:
        if not isinstance(item, dict):
            continue
        item["evidence"] = flatten_evidence(item.get("evidence"))
        if isinstance(item["evidence"], list):
            pages = {
                entry.get("page")
                for entry in item["evidence"]
                if isinstance(entry, dict)
            }
            item["source_pages"] = sorted(
                page for page in pages if type(page) is int
            )
    return {"result": items}


def flatten_evidence(evidence):
    """Turn {"halaman_55": {...}, "bukti_lain": [...]} into a plain list."""
    if not isinstance(evidence, dict):
        return evidence
    entries = []
    for key, value in evidence.items():
        match = re.fullmatch(r"halaman_(\d+)", key)
        if match and isinstance(value, dict):
            entries.append({"page": int(match.group(1)), **value})
        elif key == "bukti_lain" and isinstance(value, list):
            entries.extend(entry for entry in value if isinstance(entry, dict))
    return entries


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


# Set from --strict. Off: only the essential checks (every question cites two
# different pages with different quotes, 3 different categories). On: also
# the evidence-kind, repeated-number and answer-number checks.
STRICT_CHECKS = False


def validate_item(item, tag, allowed):
    """Return (error or None, uses_two_or_more_pages)."""
    if not isinstance(item, dict):
        return f"{tag} must be an object", False
    for field in FIELDS:
        if field not in item:
            return f"{tag} is missing '{field}'", False
    for field in ("query", "answer"):
        if not isinstance(item[field], str) or not item[field].strip():
            return f"{tag}.{field} is empty", False

    category = item["category"]
    if category not in CATEGORIES:
        return f"{tag} has an invalid category {category!r}", False
    if item["answer_type"] not in ("pendek", "panjang"):
        return f"{tag} has an invalid answer_type", False

    pages = item["source_pages"]
    if (
        not isinstance(pages, list)
        or not pages
        or any(type(page) is not int for page in pages)
    ):
        return f"{tag}.source_pages must be a non-empty list of page numbers", False
    if not set(pages) <= allowed:
        return (
            f"{tag}.source_pages {pages} includes pages that were not "
            f"provided {sorted(allowed)}",
            False,
        )

    evidence = item["evidence"]
    if not isinstance(evidence, list) or not evidence:
        return f"{tag}.evidence must be a non-empty list", False
    quotes_by_page = defaultdict(list)
    kinds = set()
    for entry in evidence:
        if (
            not isinstance(entry, dict)
            or entry.get("page") not in allowed
            or not isinstance(entry.get("quote"), str)
            or not entry["quote"].strip()
        ):
            return (
                f"{tag}.evidence entries need a provided page and a "
                "non-empty quote",
                False,
            )
        quotes_by_page[entry["page"]].append(entry["quote"])
        kinds.add(entry.get("kind"))

    if set(pages) != set(quotes_by_page):
        return (
            f"{tag}.source_pages {sorted(set(pages))} must equal the pages "
            f"in its evidence {sorted(quotes_by_page)}",
            False,
        )

    required_kind = CATEGORIES[category][0]
    if STRICT_CHECKS and required_kind not in kinds:
        return (
            f"{tag} is '{category}' but has no '{required_kind}' evidence",
            False,
        )

    page_numbers = {
        page: number_keys(" ".join(quotes))
        for page, quotes in quotes_by_page.items()
    }
    all_numbers = set().union(*page_numbers.values())
    is_multi = len(quotes_by_page) >= 2

    if is_multi:
        page_text = {
            page: normalize_text(" ".join(quotes))
            for page, quotes in quotes_by_page.items()
        }
        for page in quotes_by_page:
            # Copying one quote into both pages is not using two pages.
            if any(
                page_text[page] == text
                for p, text in page_text.items()
                if p != page
            ):
                return (
                    f"{tag}: page {page} repeats the same quote as another "
                    "page",
                    True,
                )
            other_numbers = set().union(
                *(n for p, n in page_numbers.items() if p != page)
            )
            if (
                STRICT_CHECKS
                and page_numbers[page]
                and page_numbers[page] <= other_numbers
            ):
                return (
                    f"{tag}: the evidence from page {page} only repeats "
                    "numbers already quoted from another page, so that page "
                    "adds nothing",
                    True,
                )

    if not STRICT_CHECKS:
        return None, is_multi
    if category in CALCULATION_CATEGORIES:
        if len(all_numbers) < 2:
            return (
                f"{tag} is a calculation but its evidence quotes fewer than "
                "two numbers",
                is_multi,
            )
    else:
        answer_numbers = number_map(item["answer"])
        missing = [
            token
            for key, token in answer_numbers.items()
            if key not in all_numbers
        ]
        if missing:
            return (
                f"{tag}: the answer uses numbers {missing} that do not "
                "appear in its evidence",
                is_multi,
            )
    return None, is_multi


def validate_qa(qa, allowed_pages, min_multi):
    result = qa.get("result")
    if not isinstance(result, list) or len(result) != 3:
        count = len(result) if isinstance(result, list) else 0
        return False, f"Expected 3 questions, got {count}"

    allowed = set(allowed_pages)
    needed = min(min_multi, 3) if len(allowed) > 1 else 0
    categories = []
    multi_count = 0
    for index, item in enumerate(result):
        tag = f"q{index + 1}"
        error, is_multi = validate_item(item, tag, allowed)
        if error:
            return False, error
        if needed >= 3 and not is_multi:
            return (
                False,
                f"{tag} uses only page {item['source_pages'][0]}; every "
                "question must combine information from two or more pages",
            )
        categories.append(item["category"])
        multi_count += is_multi

    if len(set(categories)) != 3:
        return False, f"Duplicate categories: {categories}"
    if multi_count < needed:
        return (
            False,
            f"Only {multi_count} of 3 questions combine two or more pages; "
            f"at least {needed} must",
        )
    return True, None


def collect_items(parsed, kept, allowed, min_multi):
    """Add this attempt's valid questions to the ones kept from earlier ones.

    Returns (kept, errors). kept maps slot index -> accepted item; a slot
    that was already accepted is never replaced. A new item is accepted only
    if it passes validate_item, combines two or more pages (when every
    question must), and does not reuse an accepted category.
    """
    kept = dict(kept)
    needed = min(min_multi, 3) if len(allowed) > 1 else 0
    used = {item["category"] for item in kept.values()}
    items = parsed.get("result", [])
    errors = []
    for index in range(len(SLOTS)):
        if index in kept:
            continue
        tag = f"q{index + 1}"
        if index >= len(items):
            errors.append(f"{tag} is missing")
            continue
        item = items[index]
        error, is_multi = validate_item(item, tag, allowed)
        if not error and needed >= 3 and not is_multi:
            error = (
                f"{tag} uses only page {item['source_pages'][0]}; every "
                "question must combine information from two or more pages"
            )
        if not error and item["category"] in used:
            error = (
                f"{tag} repeats category {item['category']!r} "
                "(Duplicate categories are not allowed)"
            )
        if error:
            errors.append(error)
            continue
        kept[index] = item
        used.add(item["category"])
    return kept, errors


def record_page_kinds(page_kinds, parsed):
    """Remember which evidence kinds the model has found on each page."""
    for item in (parsed or {}).get("result", []):
        if not isinstance(item, dict) or not isinstance(item.get("evidence"), list):
            continue
        for entry in item["evidence"]:
            if isinstance(entry, dict) and entry.get("kind") in KINDS:
                kinds = page_kinds.setdefault(entry.get("page"), [])
                if entry["kind"] not in kinds:
                    kinds.append(entry["kind"])


def plan_categories(kept, parsed, pairs, page_kinds):
    """Keep accepted slots' categories; give the other slots unused ones.

    A slot only gets a category whose evidence kind was seen on one of that
    slot's two pages, so a pair of text pages is never forced to a Chart
    category. Slots with the fewest options are filled first.
    """
    used = {item["category"] for item in kept.values()}
    order = repair_categories(parsed, len(CATEGORY_NAMES))
    plan = {index: item["category"] for index, item in kept.items()}

    def options(index):
        kinds = set()
        for page in pairs[index]:
            kinds.update(page_kinds.get(page, []))
        used_kinds = {CATEGORIES[name][0] for name in used}
        fits = [
            name
            for name in order
            if name not in used and (not kinds or CATEGORIES[name][0] in kinds)
        ]
        # Prefer a kind no other slot uses yet (text, table, chart mix).
        return sorted(fits, key=lambda name: CATEGORIES[name][0] in used_kinds)

    missing = [index for index in range(len(SLOTS)) if index not in kept]
    for index in sorted(missing, key=lambda i: len(options(i))):
        choices = options(index) or [n for n in order if n not in used]
        plan[index] = choices[0]
        used.add(choices[0])
    return [plan[index] for index in range(len(SLOTS))]


def repair_categories(qa, count=3):
    """Pick different categories, preferring evidence kinds the model
    actually found on these pages (so a page with tables gets a Table
    category rather than a Chart one)."""
    chosen, kinds = [], []
    if qa:
        for item in qa.get("result", []):
            if not isinstance(item, dict):
                continue
            category = item.get("category")
            if category in CATEGORIES and category not in chosen:
                chosen.append(category)
            for entry in item.get("evidence") or []:
                kind = entry.get("kind") if isinstance(entry, dict) else None
                if kind in KINDS and kind not in kinds:
                    kinds.append(kind)
    # First one category per evidence kind seen, then the rest of those kinds.
    for kind in kinds:
        if not any(CATEGORIES[name][0] == kind for name in chosen):
            chosen.extend(
                next(
                    [name]
                    for name in CATEGORY_NAMES
                    if CATEGORIES[name][0] == kind
                )
            )
    chosen.extend(
        name
        for name in CATEGORY_NAMES
        if name not in chosen and CATEGORIES[name][0] in kinds
    )
    chosen.extend(name for name in CATEGORY_NAMES if name not in chosen)
    return chosen[:count]


def apply_forced_categories(qa, categories):
    if not categories:
        return
    for item, category in zip(qa.get("result", []), categories):
        if isinstance(item, dict) and category:
            item["category"] = category


# --------------------------------------------------------------------------
# OCR check (optional)
# --------------------------------------------------------------------------


class PageOCR:
    def __init__(self, lang):
        try:
            import pytesseract
        except ImportError:
            sys.exit(
                "ERROR: --ocr needs pytesseract (pip install pytesseract) "
                "and the tesseract binary (apt install tesseract-ocr)."
            )
        self._tesseract = pytesseract
        self.lang = lang
        self._cache = {}

    def numbers(self, path):
        key = str(path)
        if key not in self._cache:
            with Image.open(path) as image:
                text = self._tesseract.image_to_string(
                    image.convert("RGB"), lang=self.lang
                )
            self._cache[key] = number_keys(text)
        return self._cache[key]


def ocr_check(qa, path_of, ocr, min_match):
    """Check evidence numbers against OCR text of the page they cite.

    Chart evidence is skipped because chart values are read from lines, not
    printed text. The leak check runs only when no chart evidence is used.
    """
    for index, item in enumerate(qa["result"]):
        tag = f"q{index + 1}"
        has_chart = False
        needed = set()
        for entry in item["evidence"]:
            if entry.get("kind") == "Grafik":
                has_chart = True
                continue
            keys = number_keys(entry["quote"])
            if not keys:
                continue
            needed |= keys
            found = ocr.numbers(path_of[entry["page"]])
            if len(keys & found) / len(keys) < min_match:
                missing = sorted(number_map(entry["quote"])[k] for k in keys - found)
                return (
                    f"{tag}: evidence numbers {missing} were not found on "
                    f"page {entry['page']}"
                )

        pages = sorted(set(item["source_pages"]))
        if len(pages) < 2 or has_chart or not needed:
            continue
        for page in pages:
            if needed <= ocr.numbers(path_of[page]):
                return (
                    f"{tag}: every number it uses appears on page {page}, so "
                    "that page alone can answer it"
                )
    return None


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

_STRUCTURED_WARNING_SHOWN = False


def strip_enums(schema):
    if isinstance(schema, dict):
        return {k: strip_enums(v) for k, v in schema.items() if k != "enum"}
    if isinstance(schema, list):
        return [strip_enums(v) for v in schema]
    return schema


def xgrammar_safe(schema):
    """Drop "enum" when this vLLM's xgrammar check rejects it.

    Otherwise vLLM falls back to outlines, which compiles every schema from
    scratch (~30 s per batch). Pages and categories are still checked by
    validate_qa, so losing the enums only loses the hard constraint.
    """
    try:
        from vllm.model_executor.guided_decoding.utils import (
            has_xgrammar_unsupported_json_features as unsupported,
        )
    except ImportError:
        return schema
    if not unsupported(schema):
        return schema
    stripped = strip_enums(schema)
    return schema if unsupported(stripped) else stripped


def sampling_params(max_tokens, stop_ids, temperature, schema=None):
    global _STRUCTURED_WARNING_SHOWN
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
    schema = xgrammar_safe(schema)
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **common, structured_outputs=StructuredOutputsParams(json=schema)
        )
    except (ImportError, TypeError):
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            **common, guided_decoding=GuidedDecodingParams(json=schema)
        )
    except (ImportError, TypeError):
        if not _STRUCTURED_WARNING_SHOWN:
            print("WARNING: structured decoding unavailable")
            _STRUCTURED_WARNING_SHOWN = True
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


def generate(llm, inputs, params, fallback_params):
    try:
        return llm.generate(inputs, params)
    except Exception as exc:
        if not is_schema_compile_error(exc):
            raise
        print(
            "WARNING: JSON schema failed to compile; retrying without "
            "constrained decoding"
        )
        return llm.generate(inputs, fallback_params)


def make_job(key, doc, pages):
    return {
        "key": key,
        "doc": doc,
        "pages": pages,
        "temperature": 0.0,
        "categories": None,
        "feedback": None,
        # Slot index -> question accepted in an earlier attempt.
        "kept": {},
        # Page -> evidence kinds the model has found there, over attempts.
        "page_kinds": {},
    }


def run_batch(llm, processor, jobs, args, stops, seen_by_doc):
    inputs, params, fallback, submitted, errors = [], [], [], [], []
    for job in jobs:
        page_numbers = [page for page, _ in job["pages"]]
        try:
            images = [
                load_image(path, args.max_edge, args.max_pixels)
                for _, path in job["pages"]
            ]
        except Exception as exc:
            errors.append((job, f"image load failed: {exc}"))
            continue
        inputs.append(
            {
                "prompt": build_group_prompt(
                    processor,
                    page_numbers,
                    seen_by_doc[job["doc"]],
                    args.min_multi_page,
                    job["feedback"],
                    job["categories"],
                ),
                "multi_modal_data": {"image": images},
            }
        )
        params.append(
            sampling_params(
                args.max_tokens,
                stops,
                job["temperature"],
                build_schema(page_numbers, job["categories"]),
            )
        )
        fallback.append(
            sampling_params(args.max_tokens, stops, job["temperature"])
        )
        submitted.append(job)

    if not inputs:
        return [], errors
    try:
        outputs = generate(llm, inputs, params, fallback)
    except Exception as exc:
        return [], errors + [
            (job, f"generation failed: {exc}") for job in submitted
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


def answers_match(reference, probe):
    reference_numbers = number_keys(reference)
    if reference_numbers:
        return reference_numbers <= number_keys(probe)
    ratio = difflib.SequenceMatcher(
        None, normalize_text(reference), normalize_text(probe)
    ).ratio()
    return ratio >= 0.6


def probe_single_pages(llm, processor, candidates, args, stops):
    """Ask each multi-page question with one cited page at a time.

    Returns {group_key: reason} for windows with a question that one page
    alone can answer.
    """
    inputs, params, fallback, meta = [], [], [], []
    for key, candidate in candidates.items():
        path_of = dict(candidate["job"]["pages"])
        images = {}
        for index, item in enumerate(candidate["qa"]["result"]):
            pages = sorted(set(item["source_pages"]))
            if len(pages) < 2:
                continue
            for page in pages:
                if page not in images:
                    images[page] = load_image(
                        path_of[page], args.max_edge, args.max_pixels
                    )
                inputs.append(
                    {
                        "prompt": build_probe_prompt(
                            processor, page, item["query"]
                        ),
                        "multi_modal_data": {"image": [images[page]]},
                    }
                )
                params.append(sampling_params(256, stops, 0.0, PROBE_SCHEMA))
                fallback.append(sampling_params(256, stops, 0.0))
                meta.append((key, index, page))

    if not inputs:
        return {}
    outputs = generate(llm, inputs, params, fallback)

    leaks = {}
    for (key, index, page), output in zip(meta, outputs):
        if key in leaks or not output.outputs:
            continue
        try:
            data = extract_json(output.outputs[0].text)
        except ValueError:
            continue
        if not isinstance(data, dict) or data.get("answerable") is not True:
            continue
        reference = candidates[key]["qa"]["result"][index]["answer"]
        if answers_match(reference, str(data.get("answer", ""))):
            leaks[key] = (
                f"q{index + 1} can be answered from page {page} alone; it "
                "must need information from two different pages"
            )
    return leaks


# --------------------------------------------------------------------------
# Output records
# --------------------------------------------------------------------------


def normalize_query(query):
    return re.sub(r"\s+", " ", query.strip().lower())


def is_near_duplicate(query, seen_queries, threshold):
    normalized = normalize_query(query)
    return any(
        difflib.SequenceMatcher(None, normalized, previous).ratio() >= threshold
        for previous in seen_queries
    )


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


def drop_evidence(qa):
    """The quotes only steer generation; saved QA keeps the pages used."""
    return {
        "result": [
            {key: value for key, value in item.items() if key != "evidence"}
            for item in qa["result"]
        ]
    }


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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default="bi_images/multi")
    parser.add_argument("--out", default="bi_multipage_qa_v2.jsonl")
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
        "--groups",
        nargs="+",
        metavar="PAGES",
        help=(
            "Explicit page groups, one window each, e.g. --groups 38-39 45-47 "
            "or --groups 38,39 45,46,47. Overrides automatic windowing."
        ),
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
        "--write-groups",
        metavar="FILE",
        help=(
            "Write the automatic windows to FILE in --groups-file format "
            "so you can edit them, then exit"
        ),
    )
    parser.add_argument(
        "--doc",
        help="Only use documents whose name contains this text",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the page windows and exit without loading the model",
    )
    parser.add_argument(
        "--allow-single-page",
        action="store_true",
        help="Also generate (single-page) QA for isolated pages",
    )
    parser.add_argument(
        "--min-multi-page",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="How many of the 3 questions must combine two or more pages",
    )
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1003520,
        help="Qwen-VL per-image pixel cap (~1280 vision tokens at 28x28)",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--text-token-reserve", type=int, default=1536)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--dup-threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Verify evidence numbers against Tesseract OCR of each page",
    )
    parser.add_argument("--ocr-lang", default="eng")
    parser.add_argument("--ocr-min-match", type=float, default=0.8)
    parser.add_argument(
        "--single-page-check",
        action="store_true",
        help="Reject questions the model can answer from one cited page",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Also check evidence kind vs category, repeated numbers across "
            "pages, and answer numbers against the evidence"
        ),
    )
    parser.add_argument(
        "--keep-evidence",
        action="store_true",
        help="Save the evidence quotes in the output (dropped by default)",
    )
    return parser.parse_args()


def parse_group_spec(spec):
    """Turn "38-39", "45-47" or "38,39,41" into a sorted list of pages."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            if high < low:
                raise ValueError(f"bad page range {part!r}")
            pages.update(range(low, high + 1))
        else:
            pages.add(int(part))
    if not pages:
        raise ValueError(f"empty page group {spec!r}")
    return sorted(pages)


def build_explicit_units(doc_groups, specs, max_group_size, allow_single):
    units, problems = [], []
    groups = [parse_group_spec(spec) for spec in specs]
    for doc, pages in doc_groups.items():
        path_of = dict(pages)
        for wanted in groups:
            label = f"{doc}: pages {wanted[0]}-{wanted[-1]}"
            missing = [page for page in wanted if page not in path_of]
            if missing:
                problems.append(f"{label} skipped, missing page(s) {missing}")
                continue
            if len(wanted) < 2 and not allow_single:
                problems.append(
                    f"{label} skipped, a group needs 2+ pages "
                    "(or use --allow-single-page)"
                )
                continue
            if len(wanted) > max_group_size:
                problems.append(
                    f"{label} skipped, {len(wanted)} pages exceed the "
                    f"context limit of {max_group_size} pages per window"
                )
                continue
            window = [(page, path_of[page]) for page in wanted]
            units.append((group_key(doc, wanted), doc, window))
    return units, problems


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


def build_units_from_file(doc_groups, entries, max_group_size, allow_single):
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
        if exact:
            matches = exact
        if not matches:
            problems.append(f"no document matches {fragment!r}")
            continue
        if len(matches) > 1:
            problems.append(
                f"{fragment!r} matches {len(matches)} documents; using all: "
                + " | ".join(matches)
            )
        found, found_problems = build_explicit_units(
            matches, specs, max_group_size, allow_single
        )
        problems.extend(found_problems)
        for unit in found:
            if unit[0] not in seen:
                seen.add(unit[0])
                units.append(unit)
    return units, problems


def write_groups_file(path, units):
    by_doc = defaultdict(list)
    for _, doc, window in units:
        pages = [page for page, _ in window]
        if pages == list(range(pages[0], pages[-1] + 1)):
            spec = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
        else:
            spec = ",".join(map(str, pages))
        by_doc[doc].append(spec)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(
            "# One line per document: <document>: <group> <group> ...\n"
            "# A group is a range (45-47) or a list (38,39,41).\n"
            "# Delete, merge or split groups so each one holds pages whose\n"
            "# content is related. Lines starting with # are ignored.\n\n"
        )
        for doc in sorted(by_doc):
            stream.write(f"{doc}: {' '.join(by_doc[doc])}\n")


def build_units(doc_groups, group_size, overlap, max_gap, allow_single):
    units = []
    isolated = 0
    for doc, pages in doc_groups.items():
        for run in split_runs(pages, max_gap):
            if len(run) < 2 and not allow_single:
                isolated += 1
                continue
            for window in make_windows(run, group_size, overlap):
                numbers = [page for page, _ in window]
                units.append((group_key(doc, numbers), doc, window))
    return units, isolated


def main():
    args = parse_args()
    global STRICT_CHECKS
    STRICT_CHECKS = args.strict

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
            args.max_pixels,
            args.max_tokens,
            args.text_token_reserve,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if safe_group_size < 2 and not args.allow_single_page:
        sys.exit(
            "ERROR: the context fits only one page per window, so multi-page "
            "QA is impossible. Lower --max-pixels or --max-tokens, or raise "
            "--max-model-len."
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
            f"WARNING: skipping {len(skipped)} filename(s) that do not match "
            "'<document>_pNNN'."
        )
    if args.doc:
        doc_groups = {
            doc: pages
            for doc, pages in doc_groups.items()
            if args.doc.lower() in doc.lower()
        }
        if not doc_groups:
            sys.exit(f"ERROR: no document name contains {args.doc!r}")

    isolated = 0
    if args.groups and args.groups_file:
        sys.exit("ERROR: use either --groups or --groups-file, not both")
    if args.groups_file:
        try:
            entries = read_groups_file(args.groups_file)
            units, problems = build_units_from_file(
                doc_groups, entries, safe_group_size, args.allow_single_page
            )
        except (OSError, ValueError) as exc:
            sys.exit(f"ERROR: --groups-file: {exc}")
        for problem in problems:
            print(f"WARNING: {problem}")
    elif args.groups:
        try:
            units, problems = build_explicit_units(
                doc_groups,
                args.groups,
                safe_group_size,
                args.allow_single_page,
            )
        except ValueError as exc:
            sys.exit(f"ERROR: --groups: {exc}")
        for problem in problems:
            print(f"WARNING: {problem}")
    else:
        units, isolated = build_units(
            doc_groups,
            safe_group_size,
            overlap,
            args.max_page_gap,
            args.allow_single_page,
        )
    if args.write_groups:
        if args.groups_file:
            sys.exit("ERROR: --write-groups uses automatic windows; drop "
                     "--groups-file")
        write_groups_file(args.write_groups, units)
        print(
            f"Wrote {len(units)} window(s) to {args.write_groups}. Edit it, "
            f"then run with --groups-file {args.write_groups}"
        )
        return
    if isolated:
        print(
            f"NOTE: skipping {isolated} isolated page(s) with no neighbour "
            f"within {args.max_page_gap} pages (use --allow-single-page or "
            "raise --max-page-gap to include them)."
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
        f"Batch: {args.batch_size} | Group size: {safe_group_size} "
        f"(overlap {overlap}, max page gap {args.max_page_gap}) | "
        f"Vision tokens/image: <= {tokens_per_image}\n"
        f"Multi-page questions required: {args.min_multi_page}/3 | "
        f"OCR check: {'on' if args.ocr else 'off'} | "
        f"Single-page check: {'on' if args.single_page_check else 'off'}\n"
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

    ocr = PageOCR(args.ocr_lang) if args.ocr else None

    print(f"Loading {args.model}...")
    llm = make_llm(args, safe_group_size)
    processor, stops = load_processor_and_stops(args.model)

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
                final_attempt = attempt == args.attempts
                if attempt > 1:
                    print(
                        f"  retry {attempt}/{args.attempts} for "
                        f"{len(jobs)} window(s)"
                    )
                results, errors = run_batch(
                    llm, processor, jobs, args, stops, seen_by_doc
                )
                retry = []

                def reject(job, message, parsed=None, kept=None):
                    last_error[job["key"]] = message
                    if kept is None:
                        kept = job["kept"]
                    # Keep categories forced by an earlier retry.
                    categories = job["categories"]
                    feedback = message
                    if kept or "Duplicate categories" in message:
                        categories = plan_categories(
                            kept,
                            parsed,
                            slot_pairs([page for page, _ in job["pages"]]),
                            job["page_kinds"],
                        )
                        # Also say it in words: the forced category is not
                        # generated, so the model only learns it from here.
                        feedback += ". Gunakan kategori: " + ", ".join(
                            f"{slot}={category}"
                            for slot, category in zip(SLOTS, categories)
                        )
                    if kept:
                        done = [SLOTS[index] for index in sorted(kept)]
                        redo = [
                            slot
                            for index, slot in enumerate(SLOTS)
                            if index not in kept
                        ]
                        feedback += (
                            f". {', '.join(done)} sudah diterima; perbaiki "
                            f"{', '.join(redo)} saja (isi slot yang sudah "
                            "diterima akan diabaikan)"
                        )
                    retry.append(
                        {
                            **job,
                            "temperature": 0.5,
                            "categories": categories,
                            "feedback": feedback,
                            "kept": kept,
                        }
                    )

                for job, message in errors:
                    reject(job, message)

                candidates = {}
                for job, output in results:
                    page_numbers = [page for page, _ in job["pages"]]
                    completion = output.outputs[0] if output.outputs else None
                    raw = completion.text.strip() if completion else ""
                    last_raw[job["key"]] = raw
                    parsed = None
                    length_note = (
                        f" (hit {args.max_tokens}-token output limit)"
                        if completion and completion.finish_reason == "length"
                        else ""
                    )
                    try:
                        parsed = normalize_qa(extract_json(raw))
                    except Exception as exc:
                        reject(job, str(exc) + length_note)
                        continue
                    apply_forced_categories(parsed, job["categories"])
                    record_page_kinds(job["page_kinds"], parsed)

                    # Keep every question that passes; redo only the rest.
                    kept, item_errors = collect_items(
                        parsed,
                        job["kept"],
                        set(page_numbers),
                        args.min_multi_page,
                    )
                    if item_errors:
                        reject(
                            job,
                            "; ".join(item_errors) + length_note,
                            parsed,
                            kept,
                        )
                        continue
                    parsed = {"result": [kept[i] for i in range(len(SLOTS))]}
                    valid, error = validate_qa(
                        parsed, page_numbers, args.min_multi_page
                    )
                    if not valid:
                        reject(job, error, parsed, {})
                        continue

                    if ocr is not None:
                        message = ocr_check(
                            parsed,
                            dict(job["pages"]),
                            ocr,
                            args.ocr_min_match,
                        )
                        if message:
                            reject(job, message, parsed, {})
                            continue
                    candidates[job["key"]] = {"job": job, "qa": parsed}

                if args.single_page_check and candidates:
                    try:
                        leaks = probe_single_pages(
                            llm, processor, candidates, args, stops
                        )
                    except Exception as exc:
                        print(f"WARNING: single-page check failed: {exc}")
                        leaks = {}
                    for key, message in leaks.items():
                        candidate = candidates.pop(key)
                        reject(
                            candidate["job"], message, candidate["qa"], {}
                        )

                for key, candidate in candidates.items():
                    job, parsed = candidate["job"], candidate["qa"]
                    previous = seen_by_doc[job["doc"]]
                    duplicates, novel = [], []
                    for item in parsed["result"]:
                        if is_near_duplicate(
                            item["query"], previous + novel, args.dup_threshold
                        ):
                            duplicates.append(item["query"])
                        else:
                            novel.append(normalize_query(item["query"]))
                    if duplicates and not final_attempt:
                        reject(
                            job,
                            "these questions repeat earlier ones: "
                            + " | ".join(duplicates),
                            parsed,
                            {},
                        )
                        continue

                    previous.extend(novel)
                    last_error.pop(key, None)
                    ok += 1
                    record = base_record(job, attempt)
                    record.update(
                        {
                            "qa": (
                                parsed
                                if args.keep_evidence
                                else drop_evidence(parsed)
                            ),
                            "near_duplicate_queries": duplicates,
                            "checks": {
                                "strict": args.strict,
                                "min_multi_page": args.min_multi_page,
                                "ocr": ocr is not None,
                                "single_page": args.single_page_check,
                            },
                        }
                    )
                    write_record(stream, record)
                    suffix = (
                        f" ({len(duplicates)} near-duplicate warning(s))"
                        if duplicates
                        else ""
                    )
                    print(f"OK   {key}: 3/3 QA{suffix}")

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
