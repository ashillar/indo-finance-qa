"""Shared four-stage multi-page QA pipeline for Indonesian financial documents (Qwen2.5-VL, Gemma 3, InternVL2.5).

Pipeline (no LLM-as-a-judge or automated VLM evaluation passes; designed for human annotation):
  Stage 1 — Per-page extraction (1 image at a time, cached to disk):
    Feed each page image once to the VLM backend to extract structured multimodal facts:
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
    Prompt the model in text-only mode to phrase the Indonesian question and answer
    around the two paired facts and precomputed reference answer, falling back to a
    deterministic template if phrasing validation fails after retries.
"""

import argparse
import difflib
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)
DEFAULT_MAX_PIXELS = 1280 * 28 * 28

ITEM_KEYS = ("query", "category", "answer", "answer_type", "source_path")
FACT_TYPES = ("text", "table", "chart")

SLOT_CATEGORIES = {
    1: ("Text-Text Inference", "Chart-Time-Sensitive"),
    2: ("Table-Numerical Calculation", "Chart-Numerical Calculation"),
    3: ("Table-Comparison and Sorting", "Chart-Information Extraction"),
}
SLOT_OF = {cat: slot for slot, cats in SLOT_CATEGORIES.items() for cat in cats}
LONG_CATEGORIES = {"Text-Text Inference", "Chart-Time-Sensitive"}
CALC_CATEGORIES = {"Table-Numerical Calculation", "Chart-Numerical Calculation"}

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
YES_NO_QUERY_RE = re.compile(r"\bapakah\b", re.IGNORECASE)
CALC_QUERY_WORD_RE = re.compile(
    r"\b(?:berapa|berapakah|hitung|hitunglah|selisih|total|jumlah|rasio|perbandingan|persentase|rata-rata)\b",
    re.IGNORECASE,
)

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "adalah",
    "ini", "itu", "atau", "juga", "akan", "oleh", "terhadap", "sebesar", "mencapai",
    "tercatat", "tahun", "bulan", "periode", "bagaimana", "berapa", "berapakah",
    "apakah", "apa", "mengapa", "manakah", "dokumen", "laporan", "provinsi", "bank",
    "indonesia", "kantor", "perwakilan", "mengalami", "menjadi", "serta", "antara",
    "dibandingkan", "jika", "maka", "dapat", "tidak", "lebih", "kurang", "sama",
    "nilai", "jumlah", "tingkat", "perkembangan", "perbandingan", "selisih", "rata",
    "per", "atas", "pun", "lagi", "dll", "sbb", "the", "and", "of", "ytd",
    "sebagai", "secara", "masing", "lainnya", "total", "sedangkan", "sementara",
    "tw", "triwulan", "semester", "rp", "persen", "data", "angka", "sektor",
}
STOPWORDS = STOPWORDS_ID
GROWTH_WORDS = (
    "yoy", "qtq", "mtm", "ctc", "c-to-c", "pertumbuhan", "tumbuh", "inflasi",
    "kontraksi", "growth", "kenaikan", "penurunan", "andil", "pangsa", "kontribusi",
)
GENERIC_WORDS = frozenset(GROWTH_WORDS) | {
    "mencapai", "sebesar", "tercatat", "mencatat", "mencatatkan", "angka", "nilai",
    "dibandingkan", "sebelumnya", "menjadi", "meningkat", "menurun", "naik", "turun",
    "tahun", "bulan", "periode", "posisi", "jumlah", "besar", "tinggi", "rendah",
    "tertinggi", "terendah", "miliar", "triliun", "juta", "ribu", "januari", "februari",
    "maret", "april", "mei", "juni", "juli", "agustus", "september", "oktober",
    "november", "desember",
}
SCALES = {
    "ribu": 1e3, "rb": 1e3, "juta": 1e6, "jt": 1e6,
    "miliar": 1e9, "milyar": 1e9, "triliun": 1e12,
}
UNIT_ALIASES = {"rupiah": "rp", "us": "usd", "us$": "usd", "$": "usd", "jiwa": "orang"}
MONTHS = [
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
]
MONTH_ALIASES = {"agt": 8, "ags": 8, "agst": 8, "aug": 8, "may": 5, "oct": 10, "dec": 12}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

NUM_TOKEN_RE = re.compile(r"(?<![\d.,A-Z])[-\u2212]?\d+(?:[.,]\d+)*")
AMBIGUOUS_NUM_RE = re.compile(r"[-\u2212]?[1-9]\d{0,2}([.,])\d{3}")
YEAR_IN_TEXT_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
STMT_SUBJECT_RE = re.compile(
    r"^(.{4,80}?)\s+(?:mencapai|sebesar|tercatat|mencatat|mencatatkan|tumbuh|"
    r"mengalami|senilai|berada\s+pada|menjadi|mengadakan|melaksanakan|disebabkan)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Prompts & JSON Schemas
# --------------------------------------------------------------------------

EXTRACT_PROMPT = """Gambar ini adalah SATU halaman laporan ekonomi/keuangan berbahasa Indonesia.
Tugas: ekstrak fakta-fakta penting (utamakan TABEL dan GRAFIK/BAGAN terlebih dahulu, lalu teks narasi) yang TERTULIS JELAS di halaman ini.

Aturan:
1. "facts": maksimal {max_facts} fakta penting.
   PRIORITAS EKSTRAKSI:
   - Jika halaman memiliki TABEL atau GRAFIK/BAGAN/DIAGRAM, WAJIB ekstrak minimal 8-10 fakta dari tabel ("type": "table") dan/atau grafik ("type": "chart") TERLEBIH DAHULU sebelum mengekstrak narasi paragraf ("type": "text").
   - JANGAN memberi label "text" untuk angka yang berasal dari sel tabel atau grafik/bagan.
   Rincian field:
   - "type":
     * "table": data angka dari sel/baris tabel, matriks, atau ringkasan indikator (wajib memiliki angka pada "value").
     * "chart": data dari label angka yang tercetak pada grafik/bagan/diagram batang/garis/pie (wajib memiliki angka pada "value").
     * "text": pernyataan narasi/kebijakan/penyebab/tren dari paragraf teks (boleh memuat angka atau tanpa angka).
   - "subject": nama indikator, baris tabel, atau seri grafik yang spesifik (mis. "Uang Kuasi", "Pertumbuhan Kredit UMKM", "Inflasi Kelompok Makanan"), maks 12 kata. Untuk tabel/grafik, gunakan NAMA BARIS atau SERI-nya, bukan judul tabel/grafik.
   - "value": angka PERSIS seperti tertulis (mis. "9,5", "3.434,1", "-18,3") tanpa satuan; isi "" hanya jika fakta narasi ("text") tidak memiliki angka.
   - "unit": satuan (mis. "%", "% yoy", "Rp triliun", "Rp miliar", "ribu orang", "bps", "poin"); isi "" jika tidak ada.
   - "period": periode data/kejadian (mis. "Agustus 2022", "Triwulan II 2020"); isi "" jika tidak ada.
   - "statement": satu kalimat lengkap dan mandiri (maks 25 kata) dalam Bahasa Indonesia yang memuat angka "value" beserta satuannya.
2. Jangan mengarang atau menaksir tinggi batang grafik tanpa label angka. Lewati nomor halaman dan nomor tabel/grafik.

Keluarkan HANYA JSON valid:
{{"facts": [{{"type": "table|chart|text", "subject": "...", "value": "...", "unit": "...", "period": "...", "statement": "..."}}]}}"""

WRITE_PROMPT = """Susun SATU pasangan tanya-jawab multi-halaman dalam Bahasa Indonesia untuk dokumen "{doc}".

Fakta Halaman A: {fact_a}
Fakta Halaman B: {fact_b}
Kategori: {category} ({goal})
Jawaban Acuan: {reference}

Aturan:
1. "query": satu pertanyaan terbuka dalam Bahasa Indonesia yang menyebut dokumen "{doc}" dan menanyakan KEDUA topik ("{subject_a}" serta "{subject_b}"). Awali dengan Bagaimana/Berapa/Berapakah/Apa/Mengapa/Manakah. JANGAN gunakan kata "Apakah".
2. JANGAN membocorkan angka jawaban di dalam "query", dan JANGAN menyebut nomor halaman, nomor tabel, atau nomor grafik.
3. "answer": jawaban lengkap dalam Bahasa Indonesia yang memuat informasi/angka dari kedua fakta sesuai Jawaban Acuan. Jangan menambah angka baru di luar fakta.
{feedback}
Keluarkan HANYA JSON valid: {{"query": "...", "answer": "..."}}"""

CATEGORY_GOALS = {
    "Text-Text Inference": "jelaskan keterkaitan atau sintesis kedua fakta naratif dari dua halaman",
    "Chart-Time-Sensitive": "jelaskan perkembangan nilai indikator grafik antarperiode",
    "Table-Numerical Calculation": "hitung selisih antara kedua nilai indikator (gunakan kata 'selisih')",
    "Chart-Numerical Calculation": "hitung selisih antara kedua nilai pada grafik (gunakan kata 'selisih')",
    "Table-Comparison and Sorting": "bandingkan manakah yang lebih tinggi di antara kedua indikator",
    "Chart-Information Extraction": "sebutkan dan bandingkan nilai kedua indikator pada grafik",
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "subject": {"type": "string"},
                    "value": {"type": "string"},
                    "unit": {"type": "string"},
                    "period": {"type": "string"},
                    "statement": {"type": "string"},
                },
                "required": ["type", "subject", "value", "unit", "period", "statement"],
            },
        },
    },
    "required": ["facts"],
}

WRITE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["query", "answer"],
}

EXTRACT_VERSION = hashlib.sha1(
    (EXTRACT_PROMPT + json.dumps(EXTRACT_SCHEMA, sort_keys=True)).encode()
).hexdigest()[:8]


# --------------------------------------------------------------------------
# Helpers: Text, JSON, Paths, Page Grouping, and Image Loading
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


def strip_markdown_fences(text):
    """Strip markdown code fences before checking for a leading '{'."""
    return re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).strip()


def parse_json(text):
    """Parse JSON from model output, safely handling both markdown fences and '{' prefill."""
    cleaned = strip_markdown_fences(text)
    if not cleaned:
        raise ValueError("no JSON object found in model output")
    if not cleaned.startswith("{") and ('"facts"' in cleaned or '"query"' in cleaned):
        first_brace = cleaned.find("{")
        first_quote = cleaned.find('"')
        if first_quote >= 0 and (first_brace < 0 or first_quote < first_brace):
            cleaned = "{" + cleaned[first_quote:]
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
        if not matches and not quiet:
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


def load_image(path, max_edge, max_pixels, align_factor=None, min_pixels=0):
    """Load an RGB page image capped at `max_edge` and `max_pixels` (aligned to `align_factor` when set)."""
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_edge / max(width, height))
    if width * height * scale**2 > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    if align_factor and align_factor > 1:
        new_w = max(align_factor, round(new_w / align_factor) * align_factor)
        new_h = max(align_factor, round(new_h / align_factor) * align_factor)
        if new_w * new_h > max_pixels:
            beta = math.sqrt((width * height) / max_pixels)
            new_w = max(align_factor, math.floor(width / beta / align_factor) * align_factor)
            new_h = max(align_factor, math.floor(height / beta / align_factor) * align_factor)
        elif min_pixels and new_w * new_h < min_pixels:
            beta = math.sqrt(min_pixels / (width * height))
            new_w = max(align_factor, math.ceil(width * beta / align_factor) * align_factor)
            new_h = max(align_factor, math.ceil(height * beta / align_factor) * align_factor)
    if (new_w, new_h) != (width, height):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


# --------------------------------------------------------------------------
# Numbers, Units, and Topics
# --------------------------------------------------------------------------


def parse_number(token):
    """Parse an Indonesian-formatted number ('3.434,1', '9,5', '-18,3') -> (value, decimals)."""
    text = (token or "").strip().replace("\u2212", "-")
    negative = text.startswith("-")
    text = text.lstrip("+-")
    if not re.fullmatch(r"\d+(?:[.,]\d+)*", text):
        return None, 0
    decimal_sep = None
    if "," in text and "." in text:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
    elif "," in text:
        decimal_sep = "," if text.count(",") == 1 else None
    elif "." in text:
        head, _, tail = text.partition(".")
        if text.count(".") == 1 and not (len(tail) == 3 and head != "0"):
            decimal_sep = "."
    if decimal_sep:
        int_part, frac = text.rsplit(decimal_sep, 1)
        value = float(f"{re.sub(r'[.,]', '', int_part)}.{frac}")
        decimals = len(frac)
    else:
        value = float(re.sub(r"[.,]", "", text))
        decimals = 0
    return (-value if negative else value), decimals


def numbers_in(text):
    """Return all numbers in `text` as a list of (abs_value, decimals)."""
    found = []
    for token in NUM_TOKEN_RE.findall(str(text or "")):
        value, decimals = parse_number(token)
        if value is not None:
            found.append((abs(value), decimals))
    return found


def is_year(value, decimals):
    return decimals == 0 and 1900 <= value <= 2100


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def contains_value(numbers, target, tol=1e-6):
    return any(close(value, abs(target), tol) for value, _ in numbers)


def digit_key(token):
    digits = re.sub(r"\D", "", token or "").lstrip("0")
    return digits if len(digits) >= 3 else None


def digit_keys_in(text):
    return {key for key in map(digit_key, NUM_TOKEN_RE.findall(str(text or ""))) if key}


def is_placeholder_number(token):
    digits = re.sub(r"\D", "", token or "")
    return len(digits) >= 6 and (digits in "01234567890123456789" or len(set(digits)) == 1)


def format_id(value, decimals):
    text = f"{abs(value):,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"-{text}" if value < 0 else text


def format_term(value, decimals):
    text = format_id(value, decimals)
    return f"({text})" if value < 0 else text


def with_unit(number, unit):
    if not unit:
        return number
    if unit.startswith("%"):
        return f"{number}{unit}"
    if unit.lower().startswith("rp"):
        rest = unit[2:].strip()
        return f"Rp{number} {rest}".strip()
    return f"{number} {unit}"


def value_text(fact):
    if fact["_value"] is None:
        return ""
    return with_unit(fact["value"], fact["unit"])


def infer_unit_from_text(value_str, statement, subject=""):
    """Recover missing unit from `value_str`, `subject`, or `statement` when the model left `unit` blank."""
    combined = f"{value_str} {subject} {statement}".lower()
    if "%" in combined or re.search(r"\bpersen\b", combined):
        return "% yoy" if any(w in combined for w in ("yoy", "qtq", "mtm", "ctc")) else "%"
    if re.search(r"\b(?:bps|basis\s+poin|basis\s+points?)\b", combined):
        return "bps"
    for scale_word in ("triliun", "miliar", "milyar", "juta", "ribu"):
        if re.search(rf"\b{scale_word}\b", combined):
            if re.search(r"\b(?:rp|rupiah|idr)\b", combined):
                return f"Rp {scale_word}"
            if re.search(r"\b(?:us\$|usd|dolar|dollar)\b", combined):
                return f"{scale_word} USD"
            if re.search(r"\b(?:orang|jiwa|pekerja|penumpang|wisatawan|penduduk|tenaga\s+kerja)\b", combined):
                return f"{scale_word} orang"
            if re.search(r"\b(?:ton|rekening|unit|transaksi|barel|liter)\b", combined):
                m = re.search(r"\b(ton|rekening|unit|transaksi|barel|liter)\b", combined)
                return f"{scale_word} {m.group(1)}"
            return f"Rp {scale_word}" if scale_word in ("triliun", "miliar", "milyar") else scale_word
    if re.search(r"\b(?:rp|rupiah|idr)\b", combined):
        return "Rp"
    if re.search(r"\b(?:us\$|usd|dolar\s+as)\b", combined):
        return "USD"
    for tok in ("poin", "kali", "ton", "rekening", "unit", "orang", "jiwa"):
        if re.search(rf"\b{tok}\b", combined):
            return "orang" if tok == "jiwa" else tok
    return ""


def unit_info(fact):
    if fact["_value"] is None:
        return {"kind": "none", "base": "", "scale": 1.0}
    unit = (fact["unit"] or "").lower()
    context = f"{unit} {fact['subject']} {fact['statement']}".lower()
    if "%" in unit or "persen" in unit:
        return {"kind": "percent", "base": "%", "scale": 1.0}
    if re.search(r"\b(?:bps|basis\s+poin|basis\s+points?)\b", unit):
        return {"kind": "abs", "base": "bps", "scale": 1.0}
    scale = 1.0
    for word, factor in SCALES.items():
        if re.search(rf"\b{word}\b", unit):
            scale = factor
            break
    # Strip scale words and growth/period qualifiers (e.g. yoy, qtq, mtm, ctc, ytd) before canonicalizing base unit.
    base = re.sub(rf"\b(?:{'|'.join(SCALES)}|yoy|qtq|mtm|ctc|ytd|c\s*to\s*c)\b", " ", unit)
    base = re.sub(r"[^a-z$]+", " ", base).strip()
    if re.search(r"\b(?:rp|rupiah|idr)\b", base):
        base = "rp"
    elif re.search(r"\b(?:usd|us\$|us|dolar|dollar)\b", base):
        base = "usd"
    elif re.search(r"\b(?:orang|jiwa|pekerja|penumpang|wisatawan|penduduk|tenaga\s+kerja)\b", base):
        base = "orang"
    elif re.search(r"\b(?:poin|indeks|point)\b", base):
        base = "poin"
    else:
        base = UNIT_ALIASES.get(base, base)
    if not base and (scale > 1.0 or re.search(r"\b(?:rp|rupiah|idr)\b", context)):
        base = "rp"
    return {"kind": "abs", "base": base, "scale": scale}


def units_compatible(fa, fb):
    """True when both facts are numeric and share compatible units."""
    if fa["_value"] is None or fb["_value"] is None:
        return False
    ua, ub = fa["_unit"], fb["_unit"]
    if fa.get("_ambiguous", "") != fb.get("_ambiguous", ""):
        return False
    if fa.get("_ambiguous") and ua["scale"] != ub["scale"]:
        return False
    return ua["kind"] == ub["kind"] and ua["base"] == ub["base"]


def topic_tokens(text, exclude=frozenset()):
    tokens = re.findall(r"[a-z][a-z0-9]*", (text or "").lower())
    return {
        t for t in tokens
        if t not in STOPWORDS and t not in exclude and (len(t) >= 3 or any(c.isdigit() for c in t))
    }


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def find_month(text):
    for token in re.findall(r"[a-z]+", text):
        if token in MONTH_ALIASES:
            return MONTH_ALIASES[token]
        for i, name in enumerate(MONTHS):
            if len(token) >= 3 and name.startswith(token):
                return i + 1
    return 0


def period_key(period, default_year=None):
    text = normalize_text(period)
    month = find_month(text)
    years = YEAR_IN_TEXT_RE.findall(text)
    if not years and month:
        years = [f"20{yy}" for yy in re.findall(r"(?<!\d)(\d{2})(?!\d)", text)] or (
            [default_year] if default_year else []
        )
    if not years:
        return None
    quarter_match = re.search(r"\b(?:triwulan|tw|q)\s*-?\s*(iv|iii|ii|i|[1-4])\b", text)
    quarter = 0
    if quarter_match:
        token = quarter_match.group(1)
        quarter = ROMAN.get(token) or int(token)
    return (int(years[-1]), quarter or (month + 2) // 3, month)


# --------------------------------------------------------------------------
# Stage 1 & Stage 2: Per-Page Extraction & Deterministic Fact Cleaning
# --------------------------------------------------------------------------


def salvage_extraction(text):
    """Parse extraction JSON; if truncated, salvage every complete fact object."""
    try:
        data = parse_json(text)
        if isinstance(data, dict) and isinstance(data.get("facts"), list):
            return data
    except ValueError:
        pass
    cleaned = strip_markdown_fences(text)
    facts = []
    match = re.search(r'"facts"\s*:\s*\[', cleaned)
    if match:
        decoder, pos = json.JSONDecoder(), match.end()
        while True:
            while pos < len(cleaned) and cleaned[pos] in " \n\r\t,":
                pos += 1
            if pos >= len(cleaned) or cleaned[pos] != "{":
                break
            try:
                obj, pos = decoder.raw_decode(cleaned, pos)
            except json.JSONDecodeError:
                break
            facts.append(obj)
    return {"facts": facts}


def infer_statement_subject(statement, fallback):
    """Extract the specific indicator/row subject from the beginning of `statement`."""
    match = STMT_SUBJECT_RE.match((statement or "").strip())
    if not match:
        return fallback
    cand = match.group(1).strip(" ,.;:-")
    if 1 <= len(cand.split()) <= 12 and topic_tokens(cand):
        return cand
    return fallback


CHART_CUE_RE = re.compile(
    r"\b(?:grafik|bagan|diagram|chart|kurva|batang|pie|visualisasi|tren|perkembangan|histori|historis)\b",
    re.IGNORECASE,
)
TABLE_CUE_RE = re.compile(
    r"\b(?:tabel|matriks|rincian|komponen|daftar|pangsa|proporsi|distribusi|sektor|"
    r"lapangan\s+usaha|pengeluaran|kabupaten|kota|bank|kredit|dana\s+pihak\s+ketiga|dpk|"
    r"inflasi|ihk|ekspor|impor|apbd|apbn|belanja|pendapatan|aset|npl|npf|car|roa|roe|"
    r"bopo|ldr|fdr|giro|tabungan|deposito|uang\s+kuasi|indikator)\b",
    re.IGNORECASE,
)
QUALITATIVE_ROW_RE = re.compile(
    r"\b(?:kegiatan|acara|program|agenda|jadwal|sosialisasi|rapat|koordinasi)\b",
    re.IGNORECASE,
)


def extract_value_from_statement(statement):
    """Recover a primary non-year numeric token from `statement` when `value` was left empty."""
    if not statement:
        return ""
    cleaned = FIG_TABLE_NUM_RE.sub(" ", statement)
    for tok in NUM_TOKEN_RE.findall(cleaned):
        if is_placeholder_number(tok):
            continue
        val, dec = parse_number(tok)
        if val is None or is_year(abs(val), dec):
            continue
        return tok
    return ""


def infer_visual_type(fact_type, subject, statement, value, unit, period):
    """Classify or promote numeric facts into 'table' or 'chart' when VLM labeled them as 'text'."""
    if value is None:
        return "text"
    if fact_type in ("table", "chart"):
        return fact_type
    combined = f"{subject} {statement}"
    if CHART_CUE_RE.search(combined):
        return "chart"
    if TABLE_CUE_RE.search(combined):
        return "table"
    return fact_type


def clean_facts(raw_facts, page):
    """Deterministically clean and deduplicate extracted facts on a single page."""
    facts = []
    for raw in raw_facts or []:
        if not isinstance(raw, dict):
            continue
        fact_type = str(raw.get("type") or "").strip().lower()
        subject = re.sub(r"\s+", " ", str(raw.get("subject") or "")).strip(" .")
        value_str = str(raw.get("value") or "").strip()
        unit = re.sub(r"\s+", " ", str(raw.get("unit") or "")).strip()
        period = re.sub(r"\s+", " ", str(raw.get("period") or "")).strip(" .")
        statement = re.sub(r"\s+", " ", str(raw.get("statement") or "")).strip()

        if fact_type not in FACT_TYPES or not subject or not statement:
            continue
        if len(subject.split()) > 15 or len(statement.split()) > 45:
            continue

        tokens = NUM_TOKEN_RE.findall(value_str)
        if not tokens and (fact_type in ("table", "chart") or infer_unit_from_text("", statement, subject)):
            recovered_tok = extract_value_from_statement(statement)
            if recovered_tok:
                tokens = [recovered_tok]
                value_str = recovered_tok
        has_num = bool(tokens)

        if has_num:
            if is_placeholder_number(tokens[0]):
                continue
            value, decimals = parse_number(tokens[0])
            if value is None:
                continue
            if not unit:
                unit = infer_unit_from_text(value_str, statement, subject)
            if is_year(abs(value), decimals) and not unit:
                continue
            digits = digit_key(tokens[0])
            in_stmt = contains_value(numbers_in(statement), value) or (
                digits and digits in digit_keys_in(statement)
            )
            # Only demote small un-united integers (1..30) when they are row numbers of a qualitative activity table.
            if (
                not in_stmt
                and not unit
                and decimals == 0
                and 1 <= abs(value) <= 30
                and len(statement.split()) >= 6
                and not numbers_in(statement)
                and QUALITATIVE_ROW_RE.search(f"{subject} {statement}")
            ):
                fact_type = "text"
                value, decimals, digits, formatted_val, ambiguous = None, 0, None, "", ""
            else:
                formatted_val = format_id(value, decimals)
                if not in_stmt:
                    p_txt = (
                        f" pada {period}"
                        if period and normalize_text(period) not in normalize_text(subject)
                        else ""
                    )
                    statement = f"{subject}{p_txt} tercatat sebesar {with_unit(formatted_val, unit)}."
                ambiguous_match = AMBIGUOUS_NUM_RE.fullmatch(tokens[0])
                ambiguous = ambiguous_match.group(1) if ambiguous_match else ""
                fact_type = infer_visual_type(fact_type, subject, statement, value, unit, period)
        else:
            # Qualitative row or narrative statement without a number in `value`.
            if len(statement.split()) < 6:
                continue
            if any(is_placeholder_number(t) for t in NUM_TOKEN_RE.findall(statement)):
                continue
            fact_type = "text"
            value, decimals, digits, formatted_val, ambiguous = None, 0, None, "", ""

        topic = topic_tokens(subject) | (topic_tokens(statement) if value is None else set())
        if not topic:
            continue

        fact = {
            "id": f"p{page}#{len(facts)}",
            "page": page,
            "type": fact_type,
            "subject": subject,
            "value": formatted_val,
            "unit": unit if value is not None else "",
            "period": period,
            "statement": statement,
            "_value": value,
            "_dec": min(decimals, 3),
            "_digits": digits,
            "_ambiguous": ambiguous,
            "_topic": topic,
        }
        fact["_unit"] = unit_info(fact)

        # Deduplicate near-identical facts on the same page.
        is_dup = False
        for existing in facts:
            if value is not None and existing["_value"] is not None:
                if close(abs(existing["_value"]), abs(value)) and (
                    jaccard(existing["_topic"], fact["_topic"]) >= 0.5
                    or normalize_text(existing["statement"]) == normalize_text(statement)
                ):
                    if existing["type"] == "text" and fact_type in ("table", "chart"):
                        existing["type"] = fact_type
                    is_dup = True
                    break
            elif value is None and existing["_value"] is None:
                if jaccard(existing["_topic"], fact["_topic"]) >= 0.65:
                    is_dup = True
                    break
        if not is_dup:
            facts.append(fact)

    # Disambiguate generic subjects (e.g. when a table/chart title was used for multiple rows).
    def label_key(f):
        return (
            normalize_text(f["subject"]),
            normalize_text(f["period"]),
            normalize_text(f["unit"]) if f["_value"] is not None else normalize_text(f["statement"][:50]),
        )

    counts = defaultdict(int)
    for f in facts:
        counts[label_key(f)] += 1
    for f in facts:
        if counts[label_key(f)] > 1 or (f["_value"] is None and re.match(r"^tabel\b|^grafik\b|^kegiatan\b", f["subject"], re.I)):
            refined = infer_statement_subject(f["statement"], f["subject"])
            if refined != f["subject"]:
                f["subject"] = refined
                f["_topic"] = topic_tokens(refined) | (topic_tokens(f["statement"]) if f["_value"] is None else set())

    # If multiple facts still share (subject, period, unit), keep the first occurrence of each label
    # rather than wiping out every row of the table.
    seen_labels = set()
    kept = []
    for f in facts:
        k = (label_key(f), normalize_text(f["statement"][:40])) if f["statement"] else label_key(f)
        if k not in seen_labels:
            seen_labels.add(k)
            kept.append(f)

    # Ensure pages with structured numeric facts contribute to table/chart slots even if the VLM
    # defaulted all fact types to "text".
    has_visual = any(f["type"] in ("table", "chart") and f["_value"] is not None for f in kept)
    if not has_visual:
        for f in kept:
            if f["_value"] is not None and (f["unit"] or f["period"]):
                if f["period"] and any(w in f"{f['subject']} {f['statement']}".lower() for w in GROWTH_WORDS):
                    f["type"] = "chart"
                else:
                    f["type"] = "table"

    for idx, f in enumerate(kept):
        f["id"] = f"p{page}#{idx}"
    return kept


def build_page_info(entry, page):
    """Build cleaned facts and number inventory for `page` from a cache entry."""
    if not entry:
        return {"path": "", "facts": [], "numbers": [], "raw_counts": []}
    raw_passes = list(entry.get("passes") or [])
    if isinstance(entry.get("facts"), list) and entry["facts"]:
        raw_passes.append({"facts": entry["facts"]})

    passes = [clean_facts(p.get("facts"), page) for p in raw_passes]
    # Merge facts across passes (richest reading first), deduplicating by value + topic.
    merged = []
    for pass_facts in sorted(passes, key=len, reverse=True):
        for fact in pass_facts:
            if any(
                (
                    fact["_value"] is not None
                    and f["_value"] is not None
                    and close(abs(f["_value"]), abs(fact["_value"]))
                    and jaccard(f["_topic"], fact["_topic"]) >= 0.5
                )
                or (
                    fact["_value"] is None
                    and f["_value"] is None
                    and jaccard(f["_topic"], fact["_topic"]) >= 0.65
                )
                for f in merged
            ):
                continue
            merged.append(dict(fact))
    for idx, fact in enumerate(merged):
        fact["id"] = f"p{page}#{idx}"

    # Collect all numbers printed on the page across statements, values, and any cached number list.
    texts = [str(t) for t in (entry.get("numbers") or [])]
    for p in raw_passes:
        texts.extend(str(t) for t in (p.get("all_numbers") or []))
        for rf in p.get("facts") or []:
            if isinstance(rf, dict):
                texts.append(str(rf.get("value") or ""))
                texts.append(str(rf.get("statement") or ""))
    numbers = [n for text in texts for n in numbers_in(text)]

    return {
        "path": entry["path"],
        "facts": merged,
        "numbers": numbers,
        "raw_counts": [len(p) for p in passes],
    }


def extract_cache_key(path, args):
    extra = ""
    if getattr(args, "pan_and_scan", False):
        extra += f"|pns{getattr(args, 'max_crops', 0)}"
    if getattr(args, "max_dynamic_patch", None) is not None:
        extra += f"|patch{args.max_dynamic_patch}"
    return f"{path}|{args.model}|{args.extract_max_pixels}{extra}|{EXTRACT_VERSION}"


def load_extract_cache(path):
    """Load extraction cache keyed by both full cache key and normalized image path."""
    cache = {}
    if not path.exists():
        return cache
    for line in path.open(encoding="utf-8"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "path" in record:
            passes = list(record.get("passes") or ([] if "facts" not in record else [{"facts": record["facts"]}]))
            existing = cache.get(str(record["path"])) or cache.get(Path(record["path"]).name)
            if existing:
                prev_passes = list(existing.get("passes") or ([] if "facts" not in existing else [{"facts": existing["facts"]}]))
                passes = prev_passes + passes
                record = dict(record, passes=passes)
                if not record.get("facts") and existing.get("facts"):
                    record["facts"] = existing["facts"]
            if "key" in record:
                cache[record["key"]] = record
            cache[str(record["path"])] = record
            cache[Path(record["path"]).name] = record
    return cache


def get_cached_entry(cache, path, args):
    entry = (
        cache.get(extract_cache_key(path, args))
        or cache.get(str(path))
        or cache.get(Path(path).name)
    )
    if entry is None:
        return None
    if entry.get("_fresh"):
        return entry
    info = build_page_info(entry, 0)
    return entry if info["facts"] else None


def load_page_for_backend(backend, path, args):
    loader = getattr(backend, "load_page_image", None)
    if callable(loader):
        return loader(path, args)
    return load_image(
        path,
        args.max_edge,
        args.extract_max_pixels,
        align_factor=getattr(backend, "image_align_factor", None),
        min_pixels=getattr(args, "extract_min_pixels", 0),
    )


def run_extraction(backend, args, paths, cache, cache_path):
    """Stage 1: Extract facts one page image at a time and cache to disk."""
    todo = [p for p in paths if get_cached_entry(cache, p, args) is None]
    if not todo:
        return
    prompt = backend.prompt(
        [{"type": "image"}, {"type": "text", "text": EXTRACT_PROMPT.format(max_facts=args.max_facts)}]
    )
    extra_info = ""
    if getattr(args, "pan_and_scan", False):
        extra_info = f", pan_and_scan=True (max_crops={args.max_crops})"
    elif getattr(args, "max_dynamic_patch", None) is not None:
        extra_info = f", max_dynamic_patch={args.max_dynamic_patch}"
    print(f"Extracting {len(todo)} page(s) (1 image at a time, max_pixels={args.extract_max_pixels}{extra_info})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as stream:
        for pos in range(0, len(todo), args.extract_chunk):
            chunk = todo[pos : pos + args.extract_chunk]
            requests = [
                (
                    {"prompt": prompt, "multi_modal_data": {"image": [load_page_for_backend(backend, path, args)]}},
                    args.extract_max_tokens,
                    0.0,
                    EXTRACT_SCHEMA,
                )
                for path in chunk
            ]
            outputs = backend.generate(requests)
            for path, (text, finish) in zip(chunk, outputs):
                parsed = salvage_extraction(text)
                parsed["truncated"] = finish == "length"
                record = {
                    "key": extract_cache_key(path, args),
                    "path": str(path),
                    "facts": parsed["facts"],
                    "passes": [parsed],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                record["_fresh"] = True
                cache[record["key"]] = record
                cache[str(path)] = record
                cache[Path(path).name] = record
            print(f"  extracted {min(pos + args.extract_chunk, len(todo))}/{len(todo)}")


# --------------------------------------------------------------------------
# Stage 3: Cross-Page Fact Pairing & Deterministic Reference Calculation
# --------------------------------------------------------------------------


def unique_to_page(fact, other):
    """True if `fact` does not appear on `other` page."""
    if fact["_value"] is not None:
        if contains_value(other["numbers"], fact["_value"]):
            return False
        if fact["_dec"] == 0 and abs(fact["_value"]) < 10:
            return not any(jaccard(fact["_topic"], g["_topic"]) >= 0.5 for g in other["facts"])
        return True
    # Pure narrative text fact: ensure no other-page fact has high topic overlap.
    return not any(
        jaccard(fact["_topic"], g["_topic"]) >= 0.5 for g in other["facts"]
    )


def describe(fact):
    text = fact["subject"]
    if fact["period"] and normalize_text(fact["period"]) not in normalize_text(text):
        text += f" pada {fact['period']}"
    return text


def capitalize(text):
    return text[:1].upper() + text[1:] if text else ""


def difference(fa, fb):
    """Compute exact difference between two compatible numeric facts in Python."""
    ua, ub = fa["_unit"], fb["_unit"]
    if ua["kind"] == "percent":
        scale, unit_word = 1.0, "poin persentase"
        a, b = fa["_value"], fb["_value"]
        decimals = min(2, max(fa["_dec"], fb["_dec"]))
    else:
        big = fa if ua["scale"] >= ub["scale"] else fb
        scale, unit_word = big["_unit"]["scale"], big["unit"]
        a = fa["_value"] * ua["scale"] / scale
        b = fb["_value"] * ub["scale"] / scale
        decimals = min(2, max(fa["_dec"], fb["_dec"])) if ua["scale"] == ub["scale"] else 2
    if fa.get("_ambiguous"):
        decimals = max(fa["_dec"], fb["_dec"])
    hi, lo = (a, b) if a >= b else (b, a)
    return {
        "hi": hi,
        "lo": lo,
        "result": round(hi - lo, decimals),
        "decimals": decimals,
        "unit_word": unit_word,
        "a": a,
        "b": b,
    }


def build_reference(category, fa, fb):
    """Build deterministic reference answer and required/allowed number lists."""
    required = []
    if fa["_value"] is not None:
        required.append((fa["_value"], 1e-6))
    if fb["_value"] is not None:
        required.append((fb["_value"], 1e-6))
    allowed = []

    if category == "Text-Text Inference":
        stmt_a = fa["statement"].rstrip(".")
        stmt_b = fb["statement"].rstrip(".")
        tail = stmt_b if stmt_b[:2].isupper() else stmt_b[:1].lower() + stmt_b[1:]
        reference = f"{stmt_a}. Sementara itu, {tail}."
        return reference, required, allowed

    if category == "Chart-Information Extraction" and not units_compatible(fa, fb):
        reference = (
            f"{capitalize(describe(fa))} tercatat {value_text(fa)}, sedangkan {describe(fb)} "
            f"tercatat {value_text(fb)}."
        )
        return reference, required, allowed

    diff = difference(fa, fb)
    result_text = with_unit(format_id(diff["result"], diff["decimals"]), diff["unit_word"])
    tol = 0.51 * 10 ** -diff["decimals"] / max(1.0, abs(diff["result"]))
    allowed += [(abs(diff["a"]), 1e-6), (abs(diff["b"]), 1e-6), (diff["result"], tol)]
    both = f"{capitalize(describe(fa))} tercatat {value_text(fa)}, sedangkan {describe(fb)} tercatat {value_text(fb)}"

    if category in CALC_CATEGORIES:
        arithmetic = f"{format_term(diff['hi'], diff['decimals'])} - {format_term(diff['lo'], diff['decimals'])}"
        reference = f"{both}, sehingga selisih keduanya adalah {arithmetic} = {result_text}."
        required.append((diff["result"], tol))
    elif category == "Table-Comparison and Sorting":
        higher = fa if diff["a"] >= diff["b"] else fb
        reference = (
            f"{both}. Dengan demikian, {describe(higher)} lebih tinggi dengan selisih "
            f"{result_text}."
        )
    elif category == "Chart-Information Extraction":
        higher = fa if diff["a"] >= diff["b"] else fb
        reference = f"{both}; nilai {describe(higher)} lebih tinggi."
    else:  # Chart-Time-Sensitive
        early, late = (fa, fb) if (fa["_pkey"] or (0, 0, 0)) <= (fb["_pkey"] or (0, 0, 0)) else (fb, fa)
        change = (diff["a"] - diff["b"]) * (1 if late is fa else -1)
        direction = "meningkat" if change > 0 else "menurun"
        reference = (
            f"{capitalize(describe(early))} tercatat {value_text(early)}, sedangkan {describe(late)} "
            f"tercatat {value_text(late)}, sehingga {direction} sebesar {result_text}."
        )
        required.append((diff["result"], tol))
    return reference, required, allowed


def build_query_template(doc_title, category, fa, fb):
    """Fallback deterministic Indonesian question when LLM phrasing fails validation."""
    da, db = describe(fa), describe(fb)
    if category == "Text-Text Inference":
        return f"Pada dokumen {doc_title}, bagaimana keterkaitan antara {da} dan {db}?"
    if category == "Chart-Time-Sensitive":
        return f"Berdasarkan dokumen {doc_title}, bagaimana perkembangan {da} dibandingkan dengan {db}?"
    if category in CALC_CATEGORIES:
        return f"Dalam dokumen {doc_title}, berapakah selisih antara {da} dan {db}?"
    if category == "Table-Comparison and Sorting":
        return f"Berdasarkan dokumen {doc_title}, bagaimana perbandingan antara {da} dan {db}, serta manakah yang lebih tinggi?"
    return f"Pada dokumen {doc_title}, berapakah nilai {da} serta {db}?"


def pair_candidates(fa, fb, doc_tokens, force_table_chart=True):
    """Return all valid category candidates for a cross-page fact pair (fa, fb)."""
    topic_a = (fa["_topic"] - doc_tokens) or fa["_topic"]
    topic_b = (fb["_topic"] - doc_tokens) or fb["_topic"]
    stmt_a = topic_tokens(fa["statement"], doc_tokens) | topic_a
    stmt_b = topic_tokens(fb["statement"], doc_tokens) | topic_b
    shared_subject = len(topic_a & topic_b)
    shared_statement = len(stmt_a & stmt_b)
    subject_sim, statement_sim = jaccard(topic_a, topic_b), jaccard(stmt_a, stmt_b)
    key_a, key_b = fa.get("_pkey"), fb.get("_pkey")
    same_period = (key_a is not None and key_a == key_b) or (
        bool(fa["period"]) and normalize_text(fa["period"]) == normalize_text(fb["period"])
    )
    diff_period = (key_a is not None and key_b is not None and key_a != key_b) or (
        bool(fa["period"])
        and bool(fb["period"])
        and normalize_text(fa["period"]) != normalize_text(fb["period"])
    )
    types = {fa["type"], fb["type"]}
    both_numeric = fa["_value"] is not None and fb["_value"] is not None

    categories = []
    related = shared_subject >= 1 or shared_statement >= 1 or same_period
    # Text-Text Inference: allows narrative text facts (with or without numbers) or cross-page synthesis.
    if "text" in types or related:
        if not both_numeric or not close(abs(fa["_value"]), abs(fb["_value"])):
            categories.append("Text-Text Inference")

    if both_numeric:
        compatible = units_compatible(fa, fb)
        distinct = not close(abs(fa["_value"]), abs(fb["_value"]))
        if compatible and distinct:
            if "chart" in types and diff_period and (related or subject_sim >= 0.15 or statement_sim >= 0.15):
                categories.append("Chart-Time-Sensitive")
            if "table" in types:
                categories += ["Table-Numerical Calculation", "Table-Comparison and Sorting"]
            if "chart" in types:
                categories += ["Chart-Numerical Calculation", "Chart-Information Extraction"]
            # Fallback when VLM labeled both numeric facts as "text" so Table-*/Chart-* slots are still filled.
            if force_table_chart and not (types & {"table", "chart"}):
                categories += ["Table-Numerical Calculation", "Table-Comparison and Sorting"]
                if diff_period and (related or subject_sim >= 0.15 or statement_sim >= 0.15):
                    categories += ["Chart-Time-Sensitive", "Chart-Numerical Calculation", "Chart-Information Extraction"]
        mixed = not compatible and distinct
        if mixed and "chart" in types and (related or force_table_chart):
            categories.append("Chart-Information Extraction")
        elif mixed and related and "Text-Text Inference" not in categories:
            categories.append("Text-Text Inference")

    base_score = 2 * subject_sim + statement_sim + (0.5 if related else 0.0) + (0.3 if same_period else 0.0)
    has_explicit_visual = bool(types & {"table", "chart"})
    both_visual = fa["type"] in ("table", "chart") and fb["type"] in ("table", "chart")
    candidates = []
    for category in dict.fromkeys(categories):
        if category == "Chart-Time-Sensitive":
            # Prioritize Chart-Time-Sensitive over Text-Text Inference in Slot 1 whenever available.
            cat_score = base_score + (0.85 if "chart" in types else 0.45)
        elif category == "Text-Text Inference":
            cat_score = base_score + (0.30 if "text" in types else 0.0)
        else:
            visual_bonus = 0.50 if both_visual else (0.30 if has_explicit_visual else -0.15)
            cat_score = base_score + visual_bonus
        reference, required, allowed = build_reference(category, fa, fb)
        candidates.append(
            {
                "category": category,
                "slot": SLOT_OF[category],
                "fa": fa,
                "fb": fb,
                "pa": fa["page"],
                "pb": fb["page"],
                "fact_ids": (fa["id"], fb["id"]),
                "reference": reference,
                "required": required,
                "allowed": allowed,
                "score": cat_score,
            }
        )
    return candidates


def set_key_tokens(facts, doc_tokens):
    exclude = doc_tokens | GENERIC_WORDS
    words = [topic_tokens(f"{f['subject']} {f['statement']}", exclude) for f in facts]
    for fact, own in zip(facts, words):
        others = set().union(*(w for w in words if w is not own)) if len(words) > 1 else set()
        subject_words = topic_tokens(fact["subject"])
        unique = [
            t for t in re.findall(r"[a-z][a-z0-9]*", f"{fact['subject']} {fact['statement']}".lower())
            if t in own and t not in others
        ]
        from_statement = [t for t in unique if t not in subject_words]
        fact["_key_tokens"] = list(dict.fromkeys(from_statement or unique))


def build_candidates(job, page_info, max_candidates, force_table_chart=True):
    doc_tokens = frozenset(topic_tokens(job["doc"].replace("_", " ").replace("-", " ")))
    doc_years = YEAR_IN_TEXT_RE.findall(job["doc"])
    if not doc_years:
        page_years = [
            y
            for info in page_info.values()
            for f in info["facts"]
            for y in YEAR_IN_TEXT_RE.findall(f"{f['period']} {f['statement']}")
        ]
        if page_years:
            doc_years = [max(set(page_years), key=page_years.count)]
    default_year = doc_years[-1] if doc_years else None
    for info in page_info.values():
        for fact in info["facts"]:
            fact["_pkey"] = period_key(fact["period"], default_year)
        set_key_tokens(info["facts"], doc_tokens)

    per_slot = {1: [], 2: [], 3: []}
    for pa, pb in itertools.combinations(job["pages"], 2):
        info_a, info_b = page_info[pa], page_info[pb]
        facts_a = [f for f in info_a["facts"] if unique_to_page(f, info_b)]
        facts_b = [f for f in info_b["facts"] if unique_to_page(f, info_a)]
        for fa in facts_a:
            for fb in facts_b:
                for cand in pair_candidates(fa, fb, doc_tokens, force_table_chart=force_table_chart):
                    per_slot[cand["slot"]].append(cand)

    for slot, cands in per_slot.items():
        cands.sort(key=lambda c: -c["score"])
        uses, kept = defaultdict(int), []
        for cand in cands:
            if any(uses[fid] >= 4 for fid in cand["fact_ids"]):
                continue
            for fid in cand["fact_ids"]:
                uses[fid] += 1
            kept.append(cand)
            if len(kept) >= max_candidates:
                break
        per_slot[slot] = kept
    return per_slot


def pair_signature(doc, cand):
    sig = []
    for f in (cand["fa"], cand["fb"]):
        key = round(abs(f["_value"]), 6) if f["_value"] is not None else normalize_text(f["subject"])
        sig.append((f["page"], key))
    return (doc, frozenset(sig))


# --------------------------------------------------------------------------
# Stage 4: Text-Only Question Writing & Deterministic Validation
# --------------------------------------------------------------------------


def fact_line(fact):
    period = f" ({fact['period']})" if fact["period"] else ""
    val = f" = {value_text(fact)}" if fact["_value"] is not None else ""
    return f"{fact['subject']}{val}{period}; pernyataan: \"{fact['statement']}\""


def named_subject(fact):
    subject_words = topic_tokens(fact["subject"])
    missing = [t for t in fact.get("_key_tokens", ()) if t not in subject_words][:2]
    if not missing:
        return fact["subject"]
    originals = [
        next((w for w in re.findall(r"\w+", fact["statement"]) if w.lower() == t), t)
        for t in missing
    ]
    return f"{fact['subject']} ({' '.join(originals)})"


def build_write_request(backend, job, cand, feedback):
    feedback_block = (
        f"\nPERBAIKI KESALAHAN SEBELUMNYA: {feedback}\n" if feedback else ""
    )
    text = WRITE_PROMPT.format(
        doc=job["doc_title"],
        fact_a=fact_line(cand["fa"]),
        fact_b=fact_line(cand["fb"]),
        category=cand["category"],
        goal=CATEGORY_GOALS[cand["category"]],
        reference=cand["reference"],
        subject_a=named_subject(cand["fa"]),
        subject_b=named_subject(cand["fb"]),
        feedback=feedback_block,
    )
    return {"prompt": backend.prompt([{"type": "text", "text": text}])}


def salvage_query(text):
    cleaned = strip_markdown_fences(text)
    match = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    if not match:
        return ""
    try:
        return re.sub(r"\s+", " ", json.loads(f'"{match.group(1)}"')).strip()
    except ValueError:
        return ""


def context_numbers(job, cand):
    texts = [job["doc_title"]]
    for fact in (cand["fa"], cand["fb"]):
        texts += [fact["period"], fact["subject"]]
    return [n for t in texts for n in numbers_in(t)]


def validate_written(job, cand, query, answer, seen_queries, batch_queries, dup_threshold):
    """Deterministically validate written query and answer. Returns (query_error, answer_error)."""
    if len(query) < 15:
        return "query is too short", None
    if YES_NO_QUERY_RE.search(query):
        return "query must not be a yes/no ('Apakah') question", None
    if PAGE_REF_RE.search(query) or FIG_TABLE_NUM_RE.search(query):
        return "query must not mention page, table, or figure numbers", None

    query_words = topic_tokens(query)
    for fact in (cand["fa"], cand["fb"]):
        wanted = fact["_topic"] | topic_tokens(fact["statement"])
        if not (wanted & query_words):
            return f"query must explicitly ask about '{fact['subject']}'", None

    if cand["category"] in CALC_CATEGORIES and not CALC_QUERY_WORD_RE.search(query):
        return "query must ask for a calculation (e.g. 'Berapa selisih ...')", None

    context = context_numbers(job, cand)
    for value, decimals in numbers_in(query):
        if is_year(value, decimals) or contains_value(context, value):
            continue
        targets = cand["required"] + cand["allowed"]
        if any(close(value, abs(t), tol) for t, tol in targets):
            return "query leaks an answer number; put numbers only in the answer", None

    if is_near_duplicate(query, seen_queries + batch_queries, dup_threshold, doc_name=job["doc"]):
        return "query repeats an existing question", None

    if len(answer) < 10 or PLACEHOLDER_ANS_RE.search(answer):
        return None, "answer is empty or a placeholder"
    if PAGE_REF_RE.search(answer):
        return None, "answer must not mention page numbers"

    answer_numbers = numbers_in(answer)
    for target, tol in cand["required"]:
        if not contains_value(answer_numbers, target, tol):
            return None, f"answer is missing required value {format_id(abs(target), 2)}"

    allowed = cand["required"] + cand["allowed"]
    allowed_numbers = (
        numbers_in(cand["fa"]["statement"])
        + numbers_in(cand["fb"]["statement"])
        + context
    )
    for value, decimals in answer_numbers:
        if is_year(value, decimals) or contains_value(allowed_numbers, value):
            continue
        if not any(close(value, abs(t), tol) for t, tol in allowed):
            return None, f"answer adds an unsupported number ({format_id(value, decimals)})"

    return None, None


# --------------------------------------------------------------------------
# Batch Orchestration & Output
# --------------------------------------------------------------------------


def pick_candidate(job, slot, doc_pairs):
    reserved = set(job["used_facts"])
    used_categories = set()
    page_pairs = set()
    used_fact_pairs = set()
    for other in (1, 2, 3):
        cand = job["current"][other] or job["filled"][other]
        if other != slot and cand:
            reserved.update(cand["fact_ids"])
            used_categories.add(cand["category"])
            page_pairs.add((cand["pa"], cand["pb"]))
            used_fact_pairs.add(frozenset(cand["fact_ids"]))
    used_values = [
        abs(f["_value"]) for f in job["used_fact_objs"] if f["_value"] is not None
    ]
    # First try strictly disjoint facts; if a page has too few facts, allow sharing at most 1 anchor fact
    # provided the pair itself is new.
    strict = [
        c for c in job["candidates"][slot]
        if c["category"] not in used_categories
        and not (set(c["fact_ids"]) & reserved)
        and pair_signature(job["doc"], c) not in doc_pairs
        and not any(
            f["_value"] is not None and any(close(abs(f["_value"]), u) for u in used_values)
            for f in (c["fa"], c["fb"])
        )
    ]
    relaxed = [
        c for c in job["candidates"][slot]
        if c["category"] not in used_categories
        and frozenset(c["fact_ids"]) not in used_fact_pairs
        and len(set(c["fact_ids"]) & reserved) <= 1
        and pair_signature(job["doc"], c) not in doc_pairs
    ]
    # Last-resort fallback for table/chart slots (2, 3) when all candidates share an anchor or doc_pair
    # from an overlapping window, ensuring Table-*/Chart-* questions are still produced.
    fallback_visual = []
    if slot in (2, 3) and not strict and not relaxed:
        fallback_visual = [
            c for c in job["candidates"][slot]
            if c["category"] not in used_categories
            and frozenset(c["fact_ids"]) not in used_fact_pairs
        ]
    viable = strict or relaxed or fallback_visual
    if not viable:
        return None
    # Balance Table-* and Chart-* across Slots 2 & 3 when both modalities exist.
    used_prefixes = {cat.split("-")[0] for cat in used_categories if cat.startswith(("Table-", "Chart-"))}
    if slot in (2, 3) and len(used_prefixes) == 1:
        wanted_prefix = "Chart-" if "Table" in used_prefixes else "Table-"
        diverse = [c for c in viable[:8] if c["category"].startswith(wanted_prefix)]
        if diverse:
            viable = diverse + [c for c in viable if c not in diverse]
    fresh = [c for c in viable[:6] if (c["pa"], c["pb"]) not in page_pairs]
    choice = (fresh or viable)[0]
    job["candidates"][slot].remove(choice)
    return choice


def accept_slot(job, slot, cand, seen_by_doc, doc_pairs):
    job["filled"][slot] = cand
    job["current"][slot] = None
    job["used_facts"].update(cand["fact_ids"])
    job["used_fact_objs"] += [cand["fa"], cand["fb"]]
    doc_pairs.add(pair_signature(job["doc"], cand))
    seen_by_doc[job["doc"]].append(normalize_text(cand["query"]))


def process_batch(backend, args, jobs, seen_by_doc, doc_pairs):
    """Run Stage 4 text-only writing across open slots in `jobs`."""
    for round_idx in range(args.rounds):
        is_last_round = round_idx == args.rounds - 1
        open_slots = []
        for job in jobs:
            # Allocate table/chart slots (2, 3) before narrative slot (1) so scarce numeric facts aren't taken first.
            for slot in (2, 3, 1):
                if job["filled"][slot]:
                    continue
                if not job["current"][slot]:
                    job["current"][slot] = pick_candidate(job, slot, doc_pairs)
                    job["tries"][slot], job["feedback"][slot] = 0, ""
                if job["current"][slot]:
                    open_slots.append((job, slot))
        if not open_slots:
            break
        for job in {id(j): j for j, _ in open_slots}.values():
            job["rounds"] += 1

        requests = [
            (
                build_write_request(backend, job, job["current"][slot], job["feedback"][slot]),
                args.write_max_tokens,
                0.0 if job["tries"][slot] == 0 else 0.4,
                WRITE_SCHEMA,
            )
            for job, slot in open_slots
        ]
        batch_queries = defaultdict(list)
        for (job, slot), (text, _) in zip(open_slots, backend.generate(requests)):
            cand = job["current"][slot]
            job["tries"][slot] += 1
            try:
                data = parse_json(text)
                query = re.sub(r"\s+", " ", str(data.get("query") or "")).strip()
                answer = re.sub(r"\s+", " ", str(data.get("answer") or "")).strip()
            except (ValueError, AttributeError) as exc:
                query = salvage_query(text)
                if query:
                    answer, a_err = "", "answer was cut off"
                    q_err = validate_written(
                        job, cand, query, cand["reference"],
                        seen_by_doc[job["doc"]], batch_queries[job["doc"]], args.dup_threshold,
                    )[0]
                else:
                    answer, q_err, a_err = "", f"invalid JSON ({exc})", None
            else:
                q_err, a_err = validate_written(
                    job, cand, query, answer,
                    seen_by_doc[job["doc"]], batch_queries[job["doc"]], args.dup_threshold,
                )

            if q_err or a_err:
                if job["tries"][slot] < args.write_tries and not is_last_round:
                    job["feedback"][slot] = q_err or a_err
                    continue
                # Retries or rounds exhausted: use template fallback so valid cross-page fact pairs are retained.
                if q_err:
                    query = build_query_template(job["doc_title"], cand["category"], cand["fa"], cand["fb"])
                    if is_near_duplicate(query, seen_by_doc[job["doc"]] + batch_queries[job["doc"]], args.dup_threshold, doc_name=job["doc"]):
                        job["last_error"] = f"q{slot}: {q_err}"
                        job["current"][slot] = None
                        continue
                answer = cand["reference"]
                cand["answer_source"] = "template"
            else:
                cand["answer_source"] = "model"

            cand["query"], cand["answer"] = query, answer
            batch_queries[job["doc"]].append(query)
            accept_slot(job, slot, cand, seen_by_doc, doc_pairs)

    # Guarantee Table-*/Chart-* slots (and any remaining open slot with viable candidates) are filled
    # via deterministic template if LLM rounds ended right after a candidate reset.
    if getattr(args, "force_table_chart", True):
        for job in jobs:
            for slot in (2, 3, 1):
                while not job["filled"][slot]:
                    cand = job["current"][slot] or pick_candidate(job, slot, doc_pairs)
                    job["current"][slot] = None
                    if not cand:
                        break
                    query = build_query_template(job["doc_title"], cand["category"], cand["fa"], cand["fb"])
                    if is_near_duplicate(query, seen_by_doc[job["doc"]], args.dup_threshold, doc_name=job["doc"]):
                        continue
                    cand["query"] = query
                    cand["answer"] = cand["reference"]
                    cand["answer_source"] = "template"
                    accept_slot(job, slot, cand, seen_by_doc, doc_pairs)
                    break


def public_fact(fact):
    return {k: fact[k] for k in ("page", "type", "subject", "value", "unit", "period", "statement")}


def write_record(stream, job, raw_json_only):
    items = []
    for slot in (1, 2, 3):
        cand = job["filled"][slot]
        if not cand:
            continue
        items.append(
            {
                "query": cand["query"],
                "category": cand["category"],
                "answer": cand["answer"],
                "answer_type": "long" if cand["category"] in LONG_CATEGORIES else "short",
                "source_path": [job["page_path"][cand["pa"]], job["page_path"][cand["pb"]]],
                "source_pages": [cand["pa"], cand["pb"]],
                "evidence": [public_fact(cand["fa"]), public_fact(cand["fb"])],
                "answer_source": cand["answer_source"],
            }
        )
    ok = len(items) >= job["min_questions"]
    if raw_json_only:
        payload = {"result": [{k: item[k] for k in ITEM_KEYS} for item in items] if ok else []}
    else:
        payload = {
            "group_key": job["key"],
            "document": job["doc"],
            "pages": job["pages"],
            "source_paths": job["source_paths"],
            "attempts": job["rounds"],
            "qa": {"result": items} if ok else None,
        }
        missing = [f"q{s}" for s in (1, 2, 3) if not job["filled"][s]]
        if missing:
            payload["missing_slots"] = missing
        if not ok:
            payload["error"] = job["last_error"] or "no valid cross-page fact pairs"
            payload["partial"] = items
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()
    return ok, len(items)


# --------------------------------------------------------------------------
# Shared vLLM Execution Helpers
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


def make_sampling_params(max_tokens, temperature, stops, schema=None, guided_backend=None, repetition_penalty=1.05):
    from vllm import SamplingParams

    common = {
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "stop_token_ids": stops,
    }
    if schema is None:
        return SamplingParams(**common)
    try:
        from vllm.sampling_params import GuidedDecodingParams

        gd_kwargs = {"json": schema}
        if guided_backend:
            gd_kwargs["backend"] = guided_backend
        return SamplingParams(**common, guided_decoding=GuidedDecodingParams(**gd_kwargs))
    except Exception:
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return SamplingParams(**common, structured_outputs=StructuredOutputsParams(json=schema))
        except Exception:
            return SamplingParams(**common)


def generate_batch(llm, stops, requests, guided_backend=None, repetition_penalty=1.05):
    if not requests:
        return []
    inputs = [req[0] for req in requests]
    has_schema = any(len(req) > 3 and req[3] is not None for req in requests)
    if has_schema:
        try:
            params = [
                make_sampling_params(m, t, stops, s, guided_backend=guided_backend, repetition_penalty=repetition_penalty)
                for _, m, t, s in requests
            ]
            outputs = llm.generate(inputs, params, use_tqdm=False)
            return [
                (o.outputs[0].text, o.outputs[0].finish_reason) if o.outputs else ("", None)
                for o in outputs
            ]
        except Exception as exc:
            if not re.search(r"schema|grammar|guided|structured|backend|xgrammar|outlines", str(exc), re.IGNORECASE):
                raise
            print(f"WARNING: guided JSON decoding unavailable ({exc}); running without schema")
    params = [make_sampling_params(m, t, stops, None, repetition_penalty=repetition_penalty) for _, m, t, _ in requests]
    outputs = llm.generate(inputs, params, use_tqdm=False)
    return [
        (o.outputs[0].text, o.outputs[0].finish_reason) if o.outputs else ("", None)
        for o in outputs
    ]


# --------------------------------------------------------------------------
# CLI & Main Pipeline Entrypoint
# --------------------------------------------------------------------------


def parse_args(backend):
    description = getattr(backend, "description", None) or (backend.__doc__ or __doc__).splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--folder", required=True, help="Folder containing page images")
    parser.add_argument("--groups-file", default=None, help="Page groups file (default: <folder>_groups.txt if present)")
    parser.add_argument("--out", help="Output JSONL path")
    parser.add_argument("--extract-cache", default=None, help="Per-page extraction cache (default: <out>.extract.jsonl)")
    parser.add_argument("--model", help="HuggingFace model ID")
    parser.add_argument("--doc", help="Filter to documents whose name contains this substring")
    parser.add_argument("--limit", type=int, help="Process at most N groups")
    parser.add_argument("--batch-size", type=int, default=8, help="Groups processed together per batch")
    parser.add_argument("--rounds", type=int, default=3, help="Max candidate writing rounds per batch")
    parser.add_argument("--write-tries", type=int, default=2, help="Text-only phrasing tries per candidate before template fallback")
    parser.add_argument("--max-candidates", type=int, default=8, help="Candidate fact pairs kept per slot")
    parser.add_argument("--min-questions", type=int, default=1, help="Minimum questions required to accept a group")
    parser.add_argument(
        "--force-table-chart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force generation and slot completion of Table-* and Chart-* QA pairs whenever numeric facts exist (default: True)",
    )
    parser.add_argument("--extract-max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Per-page pixel cap")
    parser.add_argument("--extract-max-tokens", type=int, default=1280)
    parser.add_argument("--extract-chunk", type=int, default=16, help="Pages per Stage 1 extraction batch")
    parser.add_argument("--max-facts", type=int, default=18)
    parser.add_argument("--max-edge", type=int, default=1680)
    parser.add_argument("--write-max-tokens", type=int, default=450)
    parser.add_argument("--dup-threshold", type=float, default=0.85)
    parser.add_argument("--path-mode", choices=("as-given", "resolve", "name"), default="as-given")
    parser.add_argument("--raw-json-only", action="store_true")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--max-page-gap", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="List page groups and exit without loading vLLM")
    parser.add_argument("--extract-only", action="store_true", help="Run per-page extraction, report fact counts, and exit")
    backend.add_args(parser)
    parser.set_defaults(**backend.defaults)
    return parser.parse_args()


def resolve_units(args, folder):
    docs = group_documents(folder)
    if args.doc:
        docs = {d: p for d, p in docs.items() if args.doc.lower() in d.lower()}
    groups_file = args.groups_file
    units = []
    if groups_file:
        units = build_units(docs, read_groups_file(groups_file), quiet=bool(args.doc))
    else:
        default_groups = Path(f"{folder.name}_groups.txt")
        if default_groups.exists():
            units = build_units(docs, read_groups_file(str(default_groups)), quiet=True)
            groups_file = str(default_groups) if units else None
        if not units:
            units = build_units(docs, auto_entries(docs, args.group_size, args.overlap, args.max_page_gap))
    return units, groups_file


def main(backend):
    args = parse_args(backend)
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"ERROR: folder does not exist: {folder}")
    units, groups_file = resolve_units(args, folder)
    if not units:
        sys.exit(f"ERROR: no multi-page groups found in {folder}.")

    tokens_per_image = backend.image_tokens(args)
    needed = tokens_per_image + args.extract_max_tokens + 600
    if needed > args.max_model_len:
        vision_knob = getattr(backend, "vision_knob", "--extract-max-pixels")
        sys.exit(
            f"ERROR: single-page extraction needs ~{needed} tokens but --max-model-len is {args.max_model_len}. "
            f"Raise --max-model-len or lower {vision_knob} / --extract-max-tokens."
        )

    out_path = Path(args.out)
    cache_path = Path(args.extract_cache or f"{out_path.with_suffix('')}.extract.jsonl")
    done, seen_by_doc = load_done(out_path)
    todo = [u for u in units if u[0] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"Model: {args.model} | Groups: {len(units)} | Done: {len(done)} | Todo: {len(todo)} | "
        f"Groups source: {groups_file or 'auto-window'}"
    )
    if args.dry_run:
        for key, _, window in units:
            print(f"  [{'done' if key in done else 'todo'}] {key} -> {[format_source_path(p, args.path_mode) for _, p in window]}")
        return
    if not todo:
        print("Nothing left to process.")
        return

    backend.load(args)
    cache = load_extract_cache(cache_path)

    if args.extract_only:
        page_paths = sorted({path for _, _, window in todo for _, path in window}, key=str)
        run_extraction(backend, args, page_paths, cache, cache_path)
        for key, doc, window in todo:
            infos = {page: build_page_info(get_cached_entry(cache, path, args), page) for page, path in window}
            counts = [f"p{page}: {len(info['facts'])} clean" for page, info in infos.items()]
            candidates = build_candidates(
                {"doc": doc, "pages": list(infos)},
                infos,
                args.max_candidates,
                force_table_chart=args.force_table_chart,
            )
            print(f"  {key}: {'; '.join(counts)} | candidates q1/q2/q3: {[len(candidates[s]) for s in (1, 2, 3)]}")
        return

    ok = fail = questions = 0
    start = time.time()
    doc_pairs = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as stream:
        for pos in range(0, len(todo), args.batch_size):
            batch_units = todo[pos : pos + args.batch_size]
            batch_pages = sorted({path for _, _, window in batch_units for _, path in window}, key=str)
            run_extraction(backend, args, batch_pages, cache, cache_path)

            jobs = []
            for key, doc, window in batch_units:
                page_info = {
                    page: build_page_info(get_cached_entry(cache, path, args), page)
                    for page, path in window
                }
                source_paths = [format_source_path(path, args.path_mode) for _, path in window]
                job = {
                    "key": key,
                    "doc": doc,
                    "doc_title": doc.replace("_", " "),
                    "pages": [page for page, _ in window],
                    "source_paths": source_paths,
                    "page_path": {page: sp for (page, _), sp in zip(window, source_paths)},
                    "page_info": page_info,
                    "filled": {1: None, 2: None, 3: None},
                    "current": {1: None, 2: None, 3: None},
                    "tries": {1: 0, 2: 0, 3: 0},
                    "feedback": {1: "", 2: "", 3: ""},
                    "used_facts": set(),
                    "used_fact_objs": [],
                    "rounds": 0,
                    "last_error": "",
                    "min_questions": args.min_questions,
                }
                job["candidates"] = build_candidates(
                    job,
                    page_info,
                    args.max_candidates,
                    force_table_chart=args.force_table_chart,
                )
                if not any(job["candidates"].values()):
                    counts = ", ".join(f"p{page}={len(info['facts'])}" for page, info in page_info.items())
                    job["last_error"] = f"no candidate fact pairs (facts per page: {counts})"
                jobs.append(job)

            process_batch(backend, args, jobs, seen_by_doc, doc_pairs)

            for job in jobs:
                accepted, count = write_record(stream, job, args.raw_json_only)
                questions += count if accepted else 0
                cats = ", ".join(job["filled"][s]["category"] for s in (1, 2, 3) if job["filled"][s])
                if accepted:
                    ok += 1
                    print(f"OK   {job['key']} ({count}/3): {cats}")
                else:
                    fail += 1
                    print(f"FAIL {job['key']}: {job['last_error'] or 'no valid cross-page fact pairs'}")
            processed = min(pos + args.batch_size, len(todo))
            print(
                f"[{processed}/{len(todo)}] ok={ok} fail={fail} questions={questions} | "
                f"{(time.time() - start) / processed:.1f}s per group"
            )

    print(
        f"\nDONE  OK: {ok} | Failed: {fail} | Questions: {questions} | Time: {time.time() - start:.0f}s\n"
        f"Output: {out_path} | Extraction cache: {cache_path}"
    )
