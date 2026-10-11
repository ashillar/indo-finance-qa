"""Generate multi-page QA pairs from page images with Qwen2.5-VL and vLLM.

Lean rewrite of multi_qwen.py for a single V100 and vLLM 0.9.x. Page groups
come from --groups-file. For each group:

1. Profile: every page is asked once what it holds (text, table, chart) and
   whether a table is cut off at the bottom or continues from the previous
   page. Profiles are cached in <out>.profiles.json.
2. Plan: q1, q2 and q3 each get two pages of the group and a multi-page
   pattern that fits what the profiles found:
     Tabel Bersambung     a table cut off on page N and continued on N+1
     Gabungan Grafik      charts on two different pages combined
     Teks dan Data        text on one page explains a table/chart on another
     Teks Lintas Halaman  text from two pages combined
     Bebas                any other question that needs both pages
   Each slot also gets a different category that fits its pattern.
3. Generate: the JSON schema makes the model quote BOTH pages before it
   writes the answer.
4. Validate: the two quotes must differ, a continued-table question must
   quote numbers from both parts, and --strict checks that answer numbers
   come from the quotes. --single-page-check asks each question with one
   page at a time and rejects it if one page is enough.

Rejected questions are retried with the reason; accepted ones are kept.
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
from itertools import product
from pathlib import Path

from PIL import Image

# V100: no FlashAttention 2 and no bfloat16; needs the vLLM V0 engine (<=0.9.x).
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_V1", "0")

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SLOTS = ("q1", "q2", "q3")
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)
# Qwen2.5-VL: 14px patch, 2x2 spatial merge -> 28x28 pixels per vision token.
PIXELS_PER_TOKEN = 28 * 28

# Category -> (page content it needs, description shown to the model).
CATEGORIES = {
    "Text-Text Inference": (
        "teks",
        "Ekstraksi atau penalaran berdasarkan informasi teks.",
    ),
    "Chart-Information Extraction": (
        "grafik",
        "Mengambil metrik atau ciri utama dari grafik.",
    ),
    "Chart-Numerical Calculation": (
        "grafik",
        "Perhitungan dari angka grafik, misalnya laju pertumbuhan atau total.",
    ),
    "Chart-Time-Sensitive": (
        "grafik",
        "Tren, sebab-akibat, urutan peristiwa, frekuensi, atau durasi dari "
        "grafik.",
    ),
    "Table-Numerical Calculation": (
        "tabel",
        "Perhitungan dari data tabel, misalnya laju pertumbuhan atau total.",
    ),
    "Table-Comparison and Sorting": (
        "tabel",
        "Membandingkan atau mengurutkan entitas berdasarkan kriteria tertentu.",
    ),
}
ALL_CATEGORIES = list(CATEGORIES)
TABLE_CATEGORIES = [n for n, (k, _) in CATEGORIES.items() if k == "tabel"]
CHART_CATEGORIES = [n for n, (k, _) in CATEGORIES.items() if k == "grafik"]
CALCULATION_CATEGORIES = {
    "Chart-Numerical Calculation",
    "Table-Numerical Calculation",
}

# Pattern -> (instruction, allowed categories, applies(profile_a, profile_b,
# adjacent)). Dict order is priority order.
PATTERNS = {
    "Tabel Bersambung": (
        "Tabel di halaman {a} TERPOTONG dan BERLANJUT di halaman {b}. Buat "
        "pertanyaan yang membutuhkan baris dari bagian awal tabel (halaman "
        "{a}) DAN baris dari bagian lanjutannya (halaman {b}), misalnya "
        "menjumlahkan, membandingkan, atau mengurutkan nilai dari kedua "
        "bagian. Kutip nama baris beserta angkanya dari masing-masing halaman.",
        TABLE_CATEGORIES,
        lambda a, b, adjacent: adjacent
        and a["tabel"]
        and b["tabel"]
        and (a["tabel_terpotong_bawah"] or b["tabel_lanjutan_atas"]),
    ),
    "Gabungan Grafik": (
        "Gabungkan data dari grafik di halaman {a} DAN grafik di halaman {b}: "
        "bandingkan nilai atau tren keduanya, hitung selisih atau rasio, atau "
        "kaitkan dua indikator yang berbeda. Kutip nilai yang dibaca dari "
        "masing-masing grafik.",
        CHART_CATEGORIES,
        lambda a, b, adjacent: a["grafik"] and b["grafik"],
    ),
    "Teks dan Data": (
        "Kaitkan penjelasan TEKS di salah satu halaman ({a} atau {b}) dengan "
        "angka TABEL atau GRAFIK di halaman lainnya.",
        ALL_CATEGORIES,
        lambda a, b, adjacent: (a["teks"] and (b["tabel"] or b["grafik"]))
        or (b["teks"] and (a["tabel"] or a["grafik"])),
    ),
    "Teks Lintas Halaman": (
        "Gabungkan informasi TEKS dari halaman {a} dan halaman {b}.",
        ["Text-Text Inference"],
        lambda a, b, adjacent: a["teks"] and b["teks"],
    ),
    "Bebas": (
        "Gabungkan informasi dari halaman {a} dan halaman {b}.",
        ALL_CATEGORIES,
        lambda a, b, adjacent: True,
    ),
}
PATTERN_NAMES = list(PATTERNS)

PROFILE_KEYS = (
    "teks",
    "tabel",
    "grafik",
    "tabel_terpotong_bawah",
    "tabel_lanjutan_atas",
)
# Used when profiling fails: allow every pattern except Tabel Bersambung.
DEFAULT_PROFILE = {
    "teks": True,
    "tabel": True,
    "grafik": True,
    "tabel_terpotong_bawah": False,
    "tabel_lanjutan_atas": False,
}

PROFILE_PROMPT = """
Gambar di atas adalah SATU halaman dokumen. Isi setiap kunci dengan true
atau false:
- "teks": ada paragraf teks penjelasan (bukan hanya judul atau catatan kaki).
- "tabel": ada tabel berisi angka.
- "grafik": ada grafik atau diagram berisi data.
- "tabel_terpotong_bawah": tabel di bagian bawah halaman terpotong dan
  berlanjut ke halaman berikutnya (misalnya tabel sampai ke tepi bawah tanpa
  baris total, atau ada tulisan "bersambung").
- "tabel_lanjutan_atas": tabel di bagian atas halaman adalah lanjutan dari
  halaman sebelumnya (misalnya tanpa judul tabel, header kolom diulang, atau
  ada tulisan "lanjutan").
Keluarkan HANYA JSON valid.
""".strip()

QA_PROMPT = """
Anda menerima {n_pages} gambar halaman dari SATU dokumen. Setiap gambar diberi
label [Halaman N]. Gunakan nomor label, BUKAN nomor yang tercetak di gambar.

TUGAS: buat TIGA pasangan pertanyaan dan jawaban dalam Bahasa Indonesia (q1,
q2, q3). SETIAP pertanyaan WAJIB membutuhkan informasi dari DUA halaman yang
ditentukan dan TIDAK BOLEH bisa dijawab dari satu halaman saja. Setiap halaman
harus menyumbang informasi yang BERBEDA; angka yang sama yang muncul di dua
halaman tidak dihitung sebagai lintas halaman.

PERTANYAAN:
{slot_block}

ATURAN:
1. Tanyakan fakta dari isi dokumen; sebutkan wilayah, periode, sektor,
   indikator, atau entitasnya.
2. Jangan gunakan frasa seperti "tabel ini", "grafik di atas", "halaman ini",
   "dokumen ini", "berdasarkan tabel", atau nomor tabel/grafik.
3. Jangan mengarang. Salin angka persis seperti yang terlihat, dengan kolom
   dan periode yang benar.
4. Isi "evidence" DULU: kutipan singkat yang persis dari SETIAP halaman
   (nama baris/kolom/periode beserta angkanya). Lalu tulis "answer" HANYA
   berdasarkan evidence, memakai informasi dari kedua halaman. Untuk
   perhitungan, tulis langkah singkatnya.
5. "answer_type": "pendek" untuk jawaban singkat, "panjang" bila perlu
   penjelasan.
6. Ketiga pertanyaan harus berbeda isinya.
{avoid_block}{feedback_block}
Keluarkan HANYA JSON valid dengan kunci "q1", "q2", dan "q3".
""".strip()

PROBE_PROMPT = """
Gambar di atas adalah SATU halaman dokumen. Jawab pertanyaan berikut HANYA
dengan informasi yang terlihat di halaman ini. Jika halaman ini tidak memuat
SEMUA informasi yang dibutuhkan, isi "answerable" dengan false dan "answer"
dengan "".

Pertanyaan: {query}

Keluarkan HANYA JSON valid.
""".strip()

STRING = {"type": "string"}
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "boolean"} for key in PROFILE_KEYS},
    "required": list(PROFILE_KEYS),
    "additionalProperties": False,
}
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"answerable": {"type": "boolean"}, "answer": STRING},
    "required": ["answerable", "answer"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Text and number helpers
# --------------------------------------------------------------------------

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Period labels such as "IV-12" or "III-2012" are not data.
PERIOD_RE = re.compile(r"\b(?:I{1,3}|IV)\s*-\s*\d{2,4}\b", re.IGNORECASE)
PAGE_REF_RE = re.compile(r"\bhalaman\s+\d+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def number_map(text):
    """Map digit keys to numbers in the text ("6,75" and "6.75" -> "675").

    Single digits, years and period labels are ignored.
    """
    text = PAGE_REF_RE.sub(" ", PERIOD_RE.sub(" ", text or ""))
    found = {}
    for token in NUMBER_RE.findall(text):
        digits = re.sub(r"\D", "", token)
        if YEAR_RE.fullmatch(token) or len(digits) < 2:
            continue
        found.setdefault(digits.lstrip("0") or "0", token)
    return found


def number_keys(text):
    return set(number_map(text))


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_near_duplicate(query, previous, threshold):
    query = normalize_text(query)
    return any(
        difflib.SequenceMatcher(None, query, other).ratio() >= threshold
        for other in previous
    )


def answers_match(reference, probe):
    reference_numbers = number_keys(reference)
    if reference_numbers:
        return reference_numbers <= number_keys(probe)
    ratio = difflib.SequenceMatcher(
        None, normalize_text(reference), normalize_text(probe)
    ).ratio()
    return ratio >= 0.6


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text or "", flags=re.I).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the output")
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


# --------------------------------------------------------------------------
# Files and page groups
# --------------------------------------------------------------------------


def normalize_name(name):
    """Repair names where percent encoding lost its percent sign."""
    if name.count(" 20") >= 2:
        name = name.replace(" 20", " ")
    return re.sub(r"\s+", " ", urllib.parse.unquote(name)).strip()


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
    """Turn "38-39", "45-47" or "38,39,41" into a sorted list of pages."""
    pages = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            pages.update(range(low, high + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def read_groups_file(path):
    """Return [(document name or part of it, [spec, ...]), ...]."""
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


def build_units(docs, entries):
    units, seen = [], set()
    for fragment, specs in entries:
        name = fragment.lower()
        matches = [d for d in docs if d.lower() == name] or [
            d for d in docs if name in d.lower()
        ]
        if not matches:
            print(f"WARNING: no document matches {fragment!r}")
        for doc in matches:
            for spec in specs:
                pages = parse_group_spec(spec)
                missing = [p for p in pages if p not in docs[doc]]
                if missing or len(pages) < 2:
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


def slot_pairs(pages):
    """[38, 39] -> (38, 39) x3; [38, 39, 40] -> (38, 39), (39, 40), (38, 40)."""
    adjacent = list(zip(pages, pages[1:]))
    others = [(a, b) for i, a in enumerate(pages) for b in pages[i + 2 :]]
    pairs = adjacent + others
    return [pairs[i % len(pairs)] for i in range(len(SLOTS))]


def load_image(path, max_edge, max_pixels):
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_edge / max(width, height))
    # Cap pixels here too: some vLLM versions ignore mm_processor_kwargs.
    if width * height * scale**2 > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
    if scale < 1:
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(size, Image.LANCZOS)
    return image


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def category_fits(category, a, b):
    kind = CATEGORIES[category][0]
    return a[kind] or b[kind]


def plan_slots(pairs, profiles, fixed, banned):
    """Pick (pattern, category) for each slot, with all categories different.

    fixed: slot index -> (pattern, category) of an accepted question.
    banned: slot indexes that failed twice; they fall back to "Bebas".
    Prefers many different special patterns, then higher-priority patterns,
    then a mix of text, table and chart categories.
    """

    def options(relaxed):
        result = []
        for index, (a, b) in enumerate(pairs):
            if index in fixed:
                result.append([fixed[index]])
                continue
            pa, pb = profiles[a], profiles[b]
            names = (
                ["Bebas"]
                if index in banned
                else [n for n in PATTERN_NAMES if PATTERNS[n][2](pa, pb, b == a + 1)]
            )
            result.append(
                [
                    (name, category)
                    for name in names
                    for category in PATTERNS[name][1]
                    if (relaxed and name == "Bebas") or category_fits(category, pa, pb)
                ]
            )
        return result

    for relaxed in (False, True):
        best, best_score = None, None
        for combo in product(*options(relaxed)):
            categories = [category for _, category in combo]
            if len(set(categories)) < len(categories):
                continue
            special = [name for name, _ in combo if name != "Bebas"]
            score = (
                len(set(special)),
                len(special),
                -sum(PATTERN_NAMES.index(name) for name, _ in combo),
                len({CATEGORIES[c][0] for c in categories}),
            )
            if best_score is None or score > best_score:
                best, best_score = combo, score
        if best:
            return list(best)
    raise RuntimeError("no category plan found")  # unreachable: 6 categories


# --------------------------------------------------------------------------
# Prompts and schemas
# --------------------------------------------------------------------------


def chat_prompt(processor, labels, text):
    content = []
    for label in labels:
        content.append({"type": "image"})
        if label:
            content.append({"type": "text", "text": label})
    content.append({"type": "text", "text": text})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def qa_prompt(processor, job, avoid_queries):
    lines = []
    for index, slot in enumerate(SLOTS):
        a, b = job["pairs"][index]
        if index in job["kept"]:
            lines.append(
                f"- {slot} (halaman {a} + {b}): SUDAH DITERIMA, isi singkat saja."
            )
            continue
        pattern, category = job["plan"][index]
        lines.append(
            f"- {slot} (halaman {a} + halaman {b}), pola {pattern}: "
            + PATTERNS[pattern][0].format(a=a, b=b)
            + f" Kategori: {category} ({CATEGORIES[category][1]})"
        )
    avoid_block = ""
    if avoid_queries:
        avoid_block = (
            "\nPertanyaan berikut sudah dipakai untuk dokumen ini; jangan "
            "diulang:\n" + "\n".join(f"- {q}" for q in avoid_queries[-10:]) + "\n"
        )
    feedback_block = ""
    if job["feedback"]:
        feedback_block = (
            f"\nPERCOBAAN SEBELUMNYA DITOLAK: {job['feedback']}\n"
            "Perbaiki kesalahan tersebut.\n"
        )
    text = QA_PROMPT.format(
        n_pages=len(job["pages"]),
        slot_block="\n".join(lines),
        avoid_block=avoid_block,
        feedback_block=feedback_block,
    )
    labels = [f"[Halaman {page}]" for page in job["pages"]]
    return {
        "prompt": chat_prompt(processor, labels, text),
        "multi_modal_data": {"image": [job["images"][p] for p in job["pages"]]},
    }


def qa_schema(pairs):
    def item(a, b):
        keys = [f"halaman_{a}", f"halaman_{b}"]
        # Property order matters: evidence is generated before the answer.
        return {
            "type": "object",
            "properties": {
                "query": STRING,
                "evidence": {
                    "type": "object",
                    "properties": {key: STRING for key in keys},
                    "required": keys,
                    "additionalProperties": False,
                },
                "answer": STRING,
                "answer_type": STRING,
            },
            "required": ["query", "evidence", "answer", "answer_type"],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {slot: item(*pair) for slot, pair in zip(SLOTS, pairs)},
        "required": list(SLOTS),
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------
# vLLM
# --------------------------------------------------------------------------


def load_model(args, max_pages):
    from transformers import AutoProcessor
    from vllm import LLM

    llm = LLM(
        model=args.model,
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        limit_mm_per_prompt={"image": max_pages},
        # No mm_processor_kwargs: this vLLM drops min/max_pixels with a
        # warning per request. load_image() already caps the pixels.
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


def sampling(max_tokens, temperature, stops, schema=None):
    from vllm import SamplingParams

    common = {
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "max_tokens": max_tokens,
        "stop_token_ids": stops,
    }
    if schema is None:
        return SamplingParams(**common)
    try:  # vLLM <= 0.10
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            **common, guided_decoding=GuidedDecodingParams(json=schema)
        )
    except ImportError:  # vLLM >= 0.11
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **common, structured_outputs=StructuredOutputsParams(json=schema)
        )


def generate(llm, stops, requests):
    """requests: [(input, max_tokens, temperature, schema)] -> output texts."""
    if not requests:
        return []
    inputs = [request[0] for request in requests]
    try:
        params = [sampling(m, t, stops, s) for _, m, t, s in requests]
        outputs = llm.generate(inputs, params, use_tqdm=False)
    except Exception as exc:
        if not re.search(r"schema|grammar", str(exc), re.I):
            raise
        print(f"WARNING: JSON schema rejected ({exc}); generating without it")
        params = [sampling(m, t, stops) for _, m, t, _ in requests]
        outputs = llm.generate(inputs, params, use_tqdm=False)
    return [
        (o.outputs[0].text, o.outputs[0].finish_reason) if o.outputs else ("", None)
        for o in outputs
    ]


def profile_pages(llm, processor, stops, units, args, cache_path):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    paths = list(
        dict.fromkeys(
            str(path)
            for _, _, window in units
            for _, path in window
            if str(path) not in cache
        )
    )
    if paths:
        print(f"Profiling {len(paths)} page(s)...")
    for start in range(0, len(paths), 16):
        chunk = paths[start : start + 16]
        requests = [
            (
                {
                    "prompt": chat_prompt(processor, [None], PROFILE_PROMPT),
                    "multi_modal_data": {
                        "image": [load_image(p, args.max_edge, args.max_pixels)]
                    },
                },
                128,
                0.0,
                PROFILE_SCHEMA,
            )
            for p in chunk
        ]
        for path, (text, _) in zip(chunk, generate(llm, stops, requests)):
            try:
                data = parse_json(text)
                cache[path] = {
                    key: bool(data.get(key, DEFAULT_PROFILE[key]))
                    for key in PROFILE_KEYS
                }
            except ValueError:
                cache[path] = dict(DEFAULT_PROFILE)
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def check_item(item, a, b, pattern, category, strict):
    """Return (error, cleaned item)."""
    if not isinstance(item, dict):
        return "missing", None
    query = str(item.get("query", "")).strip()
    answer = str(item.get("answer", "")).strip()
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    quote_a = str(evidence.get(f"halaman_{a}", "")).strip()
    quote_b = str(evidence.get(f"halaman_{b}", "")).strip()
    if not query or not answer:
        return "query or answer is empty", None
    if not quote_a or not quote_b:
        return f"evidence must quote both page {a} and page {b}", None
    text_a, text_b = normalize_text(quote_a), normalize_text(quote_b)
    if text_a in text_b or text_b in text_a:
        return f"the quotes from page {a} and page {b} are the same", None

    numbers_a, numbers_b = number_keys(quote_a), number_keys(quote_b)
    if pattern == "Tabel Bersambung" and not (numbers_a and numbers_b):
        return (
            "a continued-table question must quote row values from BOTH "
            f"parts of the table (page {a} and page {b})"
        ), None
    if strict:
        for page, mine, other in ((a, numbers_a, numbers_b), (b, numbers_b, numbers_a)):
            if mine and mine <= other:
                return f"page {page} only repeats numbers from the other page", None
        evidence_numbers = numbers_a | numbers_b
        if category in CALCULATION_CATEGORIES:
            if len(evidence_numbers) < 2:
                return "a calculation needs at least two quoted numbers", None
        else:
            missing = [
                token
                for key, token in number_map(answer).items()
                if key not in evidence_numbers
            ]
            if missing:
                return f"answer numbers {missing} are not in the evidence", None

    answer_type = str(item.get("answer_type", "")).strip().lower()
    if answer_type not in ("pendek", "panjang"):
        answer_type = "panjang" if len(answer) > 80 else "pendek"
    return None, {
        "query": query,
        "category": category,
        "multipage_type": pattern,
        "answer": answer,
        "answer_type": answer_type,
        "source_pages": [a, b],
        "evidence": {str(a): quote_a, str(b): quote_b},
    }


def single_page_leaks(llm, processor, stops, candidates):
    """Return {(job key, slot index): reason} for questions one page answers."""
    requests, meta = [], []
    for job, index, item in candidates:
        for page in item["source_pages"]:
            requests.append(
                (
                    {
                        "prompt": chat_prompt(
                            processor,
                            [None],
                            PROBE_PROMPT.format(query=item["query"]),
                        ),
                        "multi_modal_data": {"image": [job["images"][page]]},
                    },
                    256,
                    0.0,
                    PROBE_SCHEMA,
                )
            )
            meta.append((job["key"], index, page, item["answer"]))
    leaks = {}
    for (key, index, page, reference), (text, _) in zip(
        meta, generate(llm, stops, requests)
    ):
        try:
            data = parse_json(text)
        except ValueError:
            continue
        if data.get("answerable") is True and answers_match(
            reference, str(data.get("answer", ""))
        ):
            leaks.setdefault(
                (key, index),
                f"can be answered from page {page} alone; it must need both pages",
            )
    return leaks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", required=True)
    parser.add_argument("--groups-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--doc", help="Only documents whose name contains this")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1003520,
        help="Per-image pixel cap (~1280 vision tokens)",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--text-token-reserve", type=int, default=1536)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--dup-threshold", type=float, default=0.85)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also check repeated numbers and answer numbers against evidence",
    )
    parser.add_argument(
        "--single-page-check",
        action="store_true",
        help="Reject questions the model can answer from one page",
    )
    parser.add_argument("--keep-evidence", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the page groups and exit without loading the model",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Profile the pages, print each group's plan, and exit",
    )
    return parser.parse_args()


def load_done(path):
    done, seen = set(), defaultdict(list)
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("qa"):
                done.add(record["group_key"])
                seen[record["document"]].extend(
                    normalize_text(item["query"]) for item in record["qa"]["result"]
                )
    return done, seen


def new_job(key, doc, window):
    pages = [page for page, _ in window]
    return {
        "key": key,
        "doc": doc,
        "window": window,
        "pages": pages,
        "pairs": slot_pairs(pages),
        "images": {},
        "plan": None,
        "kept": {},
        "fails": [0] * len(SLOTS),
        "errors": {},
        "feedback": "",
        "raw": "",
        "duplicates": [],
    }


def write(stream, job, attempts, profile_of, extra):
    record = {
        "group_key": job["key"],
        "document": job["doc"],
        "pages": job["pages"],
        "source_paths": [str(path.resolve()) for _, path in job["window"]],
        "page_profiles": {page: profile_of[page] for page in job["pages"]},
        "attempts": attempts,
        **extra,
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def main():
    args = parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"ERROR: folder does not exist: {folder}")
    docs = group_documents(folder)
    if args.doc:
        docs = {d: p for d, p in docs.items() if args.doc.lower() in d.lower()}
    units = build_units(docs, read_groups_file(args.groups_file))
    if not units:
        sys.exit("ERROR: no page groups to process")

    max_pages = max(len(window) for _, _, window in units)
    tokens_per_image = args.max_pixels // PIXELS_PER_TOKEN
    needed = max_pages * tokens_per_image + args.max_tokens + args.text_token_reserve
    if needed > args.max_model_len:
        sys.exit(
            f"ERROR: a {max_pages}-page group needs ~{needed} tokens but "
            f"--max-model-len is {args.max_model_len}. Raise it or lower "
            "--max-pixels / --max-tokens."
        )

    out_path = Path(args.out)
    done, seen_by_doc = load_done(out_path)
    todo = [unit for unit in units if unit[0] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"Model: {args.model} | Groups: {len(units)} | Done: {len(done)} | "
        f"Todo: {len(todo)} | Max pages/group: {max_pages}\n"
        f"Strict: {args.strict} | Single-page check: {args.single_page_check}"
    )
    if args.dry_run:
        for key, _, _ in units:
            print(f"  [{'done' if key in done else 'todo'}] {key}")
        return
    if not todo:
        print("Nothing to process.")
        return

    llm, processor, stops = load_model(args, max_pages)
    profiles = profile_pages(
        llm,
        processor,
        stops,
        todo,
        args,
        out_path.with_name(out_path.name + ".profiles.json"),
    )

    def profile_map(job):
        return {page: profiles[str(path)] for page, path in job["window"]}

    if args.plan_only:
        for key, doc, window in todo:
            job = new_job(key, doc, window)
            plan = plan_slots(job["pairs"], profile_map(job), {}, set())
            print(key)
            for slot, pair, (pattern, category) in zip(SLOTS, job["pairs"], plan):
                print(f"  {slot} pages {pair}: {pattern} / {category}")
        return

    ok = fail = 0
    start_time = time.time()
    with out_path.open("a", encoding="utf-8") as stream:
        for position in range(0, len(todo), args.batch_size):
            jobs = [new_job(*unit) for unit in todo[position : position + args.batch_size]]
            for job in jobs:
                job["images"] = {
                    page: load_image(path, args.max_edge, args.max_pixels)
                    for page, path in job["window"]
                }

            for attempt in range(1, args.attempts + 1):
                active = [job for job in jobs if len(job["kept"]) < len(SLOTS)]
                if not active:
                    break
                final = attempt == args.attempts
                for job in active:
                    fixed = {
                        i: (item["multipage_type"], item["category"])
                        for i, item in job["kept"].items()
                    }
                    banned = {i for i, n in enumerate(job["fails"]) if n >= 2}
                    job["plan"] = plan_slots(
                        job["pairs"], profile_map(job), fixed, banned
                    )
                    job["errors"] = {}

                outputs = generate(
                    llm,
                    stops,
                    [
                        (
                            qa_prompt(processor, job, seen_by_doc[job["doc"]]),
                            args.max_tokens,
                            0.0 if attempt == 1 else 0.5,
                            qa_schema(job["pairs"]),
                        )
                        for job in active
                    ],
                )

                candidates = []
                for job, (text, finish_reason) in zip(active, outputs):
                    job["raw"] = text
                    try:
                        data = parse_json(text)
                    except ValueError as exc:
                        job["feedback"] = str(exc) + (
                            "; the output hit the token limit, keep quotes and "
                            "answers shorter"
                            if finish_reason == "length"
                            else ""
                        )
                        continue
                    for index, slot in enumerate(SLOTS):
                        if index in job["kept"]:
                            continue
                        pattern, category = job["plan"][index]
                        a, b = job["pairs"][index]
                        error, item = check_item(
                            data.get(slot), a, b, pattern, category, args.strict
                        )
                        if error:
                            job["errors"][index] = error
                        else:
                            candidates.append((job, index, item))

                if args.single_page_check and candidates:
                    leaks = single_page_leaks(llm, processor, stops, candidates)
                    for job, index, _ in candidates:
                        if (job["key"], index) in leaks:
                            job["errors"][index] = leaks[(job["key"], index)]
                    candidates = [
                        c for c in candidates if (c[0]["key"], c[1]) not in leaks
                    ]

                for job, index, item in candidates:
                    previous = seen_by_doc[job["doc"]] + [
                        normalize_text(kept["query"]) for kept in job["kept"].values()
                    ]
                    if is_near_duplicate(item["query"], previous, args.dup_threshold):
                        if not final:
                            job["errors"][index] = "repeats an earlier question"
                            continue
                        job["duplicates"].append(item["query"])
                    job["kept"][index] = item

                for job in active:
                    if len(job["kept"]) == len(SLOTS):
                        items = [job["kept"][i] for i in range(len(SLOTS))]
                        seen_by_doc[job["doc"]].extend(
                            normalize_text(item["query"]) for item in items
                        )
                        if not args.keep_evidence:
                            items = [
                                {k: v for k, v in item.items() if k != "evidence"}
                                for item in items
                            ]
                        write(
                            stream,
                            job,
                            attempt,
                            profile_map(job),
                            {
                                "qa": {"result": items},
                                "near_duplicate_queries": job["duplicates"],
                            },
                        )
                        ok += 1
                        patterns = ", ".join(item["multipage_type"] for item in items)
                        print(f"OK   {job['key']} (attempt {attempt}): {patterns}")
                        continue
                    for index in job["errors"]:
                        job["fails"][index] += 1
                    if job["errors"]:
                        job["feedback"] = "; ".join(
                            f"{SLOTS[i]}: {error}"
                            for i, error in sorted(job["errors"].items())
                        )
                    if job["kept"]:
                        job["feedback"] += (
                            ". Sudah diterima: "
                            + ", ".join(SLOTS[i] for i in sorted(job["kept"]))
                            + "; perbaiki slot lainnya saja"
                        )

            for job in jobs:
                if len(job["kept"]) < len(SLOTS):
                    fail += 1
                    write(
                        stream,
                        job,
                        args.attempts,
                        profile_map(job),
                        {
                            "qa": None,
                            "error": job["feedback"],
                            "partial": [job["kept"][i] for i in sorted(job["kept"])],
                            "raw_output": job["raw"],
                        },
                    )
                    print(f"FAIL {job['key']}: {job['feedback']}")

            processed = min(position + args.batch_size, len(todo))
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
