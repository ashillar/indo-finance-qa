import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "json"
CLUSTER_ROOT = "/home/cluster-dgx1/ashillafryda/vllm-project"

SOURCES = {
    "BI": {
        "internvl": "bi_multipage_batch_internvl.jsonl",
        "qwen": "bi_multipage_qwen25_vl.jsonl",
        "images_name": "bi_images",
        # Folders searched for the page images (first one that exists is used).
        "image_dirs": ["bi_multi", "bi-multi", "bi_images/multi", "bi_images"],
    },
    "OJK": {
        "internvl": "ojk_multipage_batch_internvl.jsonl",
        "qwen": "ojk_multipage_qwen25_vl.jsonl",
        "images_name": "ojk_images",
        "image_dirs": ["ojk_multi", "ojk-multi", "ojk_images/multi", "ojk_images"],
    },
}

# Set from the sidebar source selector before any data is loaded.
INTERNVL_JSONL = QWEN_JSONL = IMAGE_ROOT = None
IMAGE_ROOTS = []
CLUSTER_PREFIX = ""
SOURCES_IMAGES_NAME = ""
COMMENTS_PATH = BASE_DIR / "qa_comments.json"

st.set_page_config(
    page_title="Multi-Page QA Review: InternVL vs Qwen",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    .main-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        line-height: 1.3;
    }
    .main-subtitle {
        color: #6b7280;
        font-size: 12.5px;
        margin-top: 2px;
        word-break: break-all;
    }
    .model-banner {
        border-radius: 9px;
        padding: 7px 12px;
        font-weight: 700;
        font-size: 13.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .banner-internvl { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .banner-qwen { background: #f5f3ff; color: #6d28d9; border: 1px solid #ddd6fe; }
    .banner-gemma { background: #f0fdfa; color: #0f766e; border: 1px solid #99f6e4; }
    .qa-category {
        display: inline-block;
        font-size: 11px;
        color: #4b5563;
        background: #f3f4f6;
        padding: 2px 8px;
        border-radius: 6px;
        margin: 0 6px 0 4px;
    }
    .qa-question {
        font-size: 14px;
        line-height: 1.5;
        font-weight: 600;
        color: #111827;
        margin: 6px 0;
    }
    .answer-label {
        font-size: 10px;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.06em;
        margin-bottom: 3px;
    }
    .answer-box {
        display: inline-block;
        padding: 6px 10px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 6px;
        color: #047857;
        font-size: 13px;
        font-weight: 600;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .answer-type { color: #9ca3af; font-size: 11px; margin-left: 6px; }
    .image-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 6px;
        color: #1f2937;
    }
    .record-counter { text-align: center; font-size: 13px; color: #6b7280; padding-top: 6px; }
    section[data-testid="stSidebar"] { background: #f9fafb; }
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
    ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_jsonl(path_string):
    path = Path(path_string)
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


@st.cache_data
def index_images(root_strings):
    index = {}
    for root_string in root_strings:
        root = Path(root_string)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".pdf"}:
                index.setdefault(path.name, []).append(str(path))
    return index


def resolve_page_image(source_path, image_index):
    source = str(source_path)
    local = Path(source)
    if local.exists():
        return local
    name = os.path.basename(source)
    if source.startswith(CLUSTER_PREFIX):
        rel = source[len(CLUSTER_PREFIX):].lstrip("/")
        for root in (BASE_DIR / SOURCES_IMAGES_NAME, BASE_DIR):
            if (root / rel).exists():
                return root / rel
    hits = image_index.get(name, [])
    return Path(hits[0]) if hits else None


def record_pages(record):
    """List of (page_label, source_path) for a record, in page order."""
    if not record:
        return []
    paths = record.get("source_paths") or []
    pages = record.get("pages") or []
    out = []
    for position, path in enumerate(paths):
        label = pages[position] if position < len(pages) else position + 1
        out.append((label, path))
    return out


def resolve_images(record, image_index):
    """Resolved images for a record: list of (page_label, Path or None)."""
    return [
        (label, resolve_page_image(path, image_index))
        for label, path in record_pages(record)
    ]


def supabase_config():
    """Return (url, key) from Streamlit secrets, or None to use the local file."""
    try:
        cfg = st.secrets["supabase"]
        return cfg["url"].rstrip("/"), cfg["key"]
    except Exception:
        return None


def _sb_headers(key, **extra):
    return {"apikey": key, "Authorization": f"Bearer {key}", **extra}


@st.cache_data(ttl=20, show_spinner=False)
def _load_remote_comments(url, key):
    comments = {}
    start, page = 0, 1000
    while True:
        resp = requests.get(
            f"{url}/rest/v1/reviews",
            params={"select": "filename,model_key,qa_index,data"},
            headers=_sb_headers(key, Range=f"{start}-{start + page - 1}"),
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json()
        for row in rows:
            comments.setdefault(row["filename"], {}).setdefault(
                row["model_key"], {}
            )[str(row["qa_index"])] = row["data"]
        if len(rows) < page:
            return comments
        start += page


def load_comments():
    config = supabase_config()
    if config:
        try:
            return _load_remote_comments(*config)
        except requests.RequestException as exc:
            st.error(f"Could not load reviews from Supabase: {exc}")
            return {}
    if not COMMENTS_PATH.exists():
        return {}
    try:
        data = json.loads(COMMENTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


CRITERIA = [
    (
        "query_clarity",
        "Query Clarity",
        "The query should be specific and unambiguous, targeting a particular topic in a document, and avoiding vague or overly general questions.",
    ),
    (
        "answer_correctness",
        "Answer Correctness",
        "The answer must be factually correct and directly supported by the visual content. It should not include hallucinations or inferred information beyond what is presented. If calculation is involved, the answer should be accurate.",
    ),
    (
        "category_appropriateness",
        "Category Appropriateness",
        "The question should match its assigned category (e.g., table-numerical calculations). Mislabelled or ambiguous categories should lead to revision or rejection.",
    ),
    (
        "multi_page_sources",
        "Multi-Page Sources",
        "For multi-page queries, if the answer is derived from multiple pages, all referenced page sources (see CITED PAGES) must be accurately identified. Use N/A only if the item is genuinely single-page.",
    ),
]
CRITERION_CHOICES = ["Unreviewed", "Pass", "Fail", "N/A"]
DECISION_CHOICES = ["Unreviewed", "Retain", "Revise", "Discard"]


def file_review_bucket(file_entry):
    if not isinstance(file_entry, dict):
        return {}
    if (
        "internvl" not in file_entry
        and "qwen" not in file_entry
        and "gemma" not in file_entry
    ):
        return {}
    return file_entry


def save_qa_review_remote(config, filename, model_key, qa_index, payload):
    url, key = config
    match = {
        "filename": f"eq.{filename}",
        "model_key": f"eq.{model_key}",
        "qa_index": f"eq.{qa_index}",
    }
    resp = requests.get(
        f"{url}/rest/v1/reviews",
        params={"select": "data", **match},
        headers=_sb_headers(key),
        timeout=15,
    )
    resp.raise_for_status()
    rows = resp.json()
    existing = rows[0]["data"] if rows and isinstance(rows[0]["data"], dict) else {}
    existing.update(payload)
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    resp = requests.post(
        f"{url}/rest/v1/reviews",
        params={"on_conflict": "filename,model_key,qa_index"},
        headers=_sb_headers(
            key,
            **{
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
        ),
        json={
            "filename": filename,
            "model_key": model_key,
            "qa_index": int(qa_index),
            "data": existing,
        },
        timeout=15,
    )
    resp.raise_for_status()
    _load_remote_comments.clear()


def save_qa_review(filename, model_key, qa_index, payload):
    config = supabase_config()
    if config:
        save_qa_review_remote(config, filename, model_key, qa_index, payload)
        return
    comments = load_comments()
    file_entry = file_review_bucket(comments.get(filename, {}))
    comments[filename] = file_entry
    model_entry = file_entry.setdefault(model_key, {})
    if not isinstance(model_entry, dict):
        model_entry = {}
        file_entry[model_key] = model_entry
    existing = model_entry.get(str(qa_index), {})
    if not isinstance(existing, dict):
        existing = {}
    existing.update(payload)
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    model_entry[str(qa_index)] = existing
    COMMENTS_PATH.write_text(
        json.dumps(comments, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def qa_comment(comments, filename, model_key, qa_index):
    file_entry = comments.get(filename, {})
    if not isinstance(file_entry, dict):
        return ""
    model_entry = file_entry.get(model_key, {})
    if not isinstance(model_entry, dict):
        return ""
    entry = model_entry.get(str(qa_index), "")
    if isinstance(entry, dict):
        return str(entry.get("comment", ""))
    return str(entry or "")


def saved_qa_entry(comments, filename, model_key, qa_index):
    file_entry = comments.get(filename, {})
    if not isinstance(file_entry, dict):
        return {}
    model_entry = file_entry.get(model_key, {})
    if not isinstance(model_entry, dict):
        return {}
    entry = model_entry.get(str(qa_index), {})
    return entry if isinstance(entry, dict) else {}


def filename_of(record):
    return str(record.get("group_key") or "")


def group_by_file(records):
    grouped = {}
    for record in records:
        name = filename_of(record)
        if not name:
            continue
        grouped.setdefault(name, []).append(record)
    return grouped


def pick_record(attempts, prefer_success=True):
    if not attempts:
        return None
    if prefer_success:
        for record in reversed(attempts):
            qa = record.get("qa")
            if isinstance(qa, dict) and qa.get("result"):
                return record
    return attempts[-1]


def qa_items(record):
    if not record:
        return []
    qa = record.get("qa")
    if isinstance(qa, dict) and isinstance(qa.get("result"), list):
        return qa["result"]
    raw = record.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("result"), list):
                return parsed["result"]
        except json.JSONDecodeError:
            pass
    return []


def status_label(record):
    if not record:
        return "missing"
    if record.get("error"):
        return "error"
    if qa_items(record):
        return "ok"
    return "empty"


def init_widget(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


MODELS = [
    ("internvl", "InternVL", "banner-internvl"),
    ("qwen", "Qwen2.5-VL", "banner-qwen"),
]
CRITERION_BUTTONS = ["Pass", "Fail", "N/A"]
DECISIONS = ["Retain", "Revise", "Discard"]
DECISION_BADGE = {
    "Retain": ":green[● Retain]",
    "Revise": ":orange[● Revise]",
    "Discard": ":red[● Discard]",
    "Unreviewed": ":gray[○ Unreviewed]",
}


def autosave_review(prefix, filename, model_key, number, query):
    payload = {
        "query": query,
        "comment": str(st.session_state.get(f"comment-{prefix}") or "").strip(),
        "decision": st.session_state.get(f"decision-{prefix}") or "Unreviewed",
        "criteria": {
            field: st.session_state.get(f"crit-{prefix}-{field}") or "Unreviewed"
            for field, _, _ in CRITERIA
        },
    }
    try:
        save_qa_review(filename, model_key, number, payload)
    except Exception as exc:  # network or disk problem: show it instead of failing silently
        st.session_state["_save_error"] = f"Could not save review: {exc}"
        return
    st.session_state.pop("_save_error", None)
    st.session_state[f"saved-{prefix}"] = datetime.now().strftime("%H:%M:%S")


def set_all_pass(prefix, filename, model_key, number, query):
    for field, _, _ in CRITERIA:
        st.session_state[f"crit-{prefix}-{field}"] = "Pass"
    autosave_review(prefix, filename, model_key, number, query)


def render_qa_cards(items, filename, model_key, comments):
    if not items:
        st.info("No Q&A parsed for this record.")
        return
    for number, qa in enumerate(items, start=1):
        if not isinstance(qa, dict):
            continue
        query = str(qa.get("query", ""))
        answer = html.escape(str(qa.get("answer", "")))
        category = html.escape(str(qa.get("category", "Unknown")))
        answer_type = html.escape(str(qa.get("answer_type", "")))
        source_pages = qa.get("source_pages")
        if isinstance(source_pages, list):
            source_pages_text = ", ".join(str(p) for p in source_pages) or "none"
        else:
            source_pages_text = str(source_pages or "none")
        existing = saved_qa_entry(comments, filename, model_key, number)
        saved_criteria = existing.get("criteria") if isinstance(existing.get("criteria"), dict) else {}
        prefix = f"{filename}-{model_key}-{number}"
        args = (prefix, filename, model_key, number, query)

        decision_key = f"decision-{prefix}"
        comment_key = f"comment-{prefix}"
        saved_decision = existing.get("decision")
        init_widget(decision_key, saved_decision if saved_decision in DECISIONS else None)
        init_widget(comment_key, qa_comment(comments, filename, model_key, number))
        for field, _, _ in CRITERIA:
            saved_value = saved_criteria.get(field)
            init_widget(
                f"crit-{prefix}-{field}",
                saved_value if saved_value in CRITERION_BUTTONS else None,
            )
        current = st.session_state.get(decision_key) or "Unreviewed"

        with st.container(border=True):
            st.markdown(
                f"**Q{number}** <span class='qa-category'>{category}</span> {DECISION_BADGE[current]}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="qa-question">{html.escape(query)}</div>',
                unsafe_allow_html=True,
            )
            if answer:
                type_html = f'<span class="answer-type">{answer_type}</span>' if answer_type else ""
                st.markdown(
                    f'<div class="answer-label">ANSWER</div><span class="answer-box">{answer}</span>{type_html}',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No answer in this object.")
            st.markdown(
                f'<div class="answer-label">CITED PAGES</div><span class="answer-box" style="background:#eff6ff;border-color:#bfdbfe;color:#1d4ed8;">pages: {html.escape(source_pages_text)}</span>',
                unsafe_allow_html=True,
            )

            st.segmented_control(
                "Decision",
                DECISIONS,
                key=decision_key,
                label_visibility="collapsed",
                on_change=autosave_review,
                args=args,
                help="Retain if all criteria are met. Revise for minor issues. Discard for major errors. Click again to clear.",
            )

            st.button(
                "✓ Mark all criteria Pass",
                key=f"allpass-{prefix}",
                on_click=set_all_pass,
                args=args,
            )
            for field, label, help_text in CRITERIA:
                label_col, control_col = st.columns([2, 3], vertical_alignment="center")
                label_col.markdown(f"<span style='font-size:13px'>{label}</span>", unsafe_allow_html=True, help=help_text)
                with control_col:
                    st.segmented_control(
                        label,
                        CRITERION_BUTTONS,
                        key=f"crit-{prefix}-{field}",
                        label_visibility="collapsed",
                        on_change=autosave_review,
                        args=args,
                    )
            st.text_area(
                "Comment",
                height=70,
                key=comment_key,
                placeholder="Optional notes: what to revise, why it is discarded, etc.",
                on_change=autosave_review,
                args=args,
            )

            stamp = st.session_state.get(f"saved-{prefix}") or existing.get("updated_at")
            if stamp:
                st.caption(f"✓ Saved {str(stamp)[:19].replace('T', ' ')}")


def render_source_images(page_images, zoom_level=100):
    for label, image_path in page_images:
        st.markdown(f"**Page {label}**")
        if image_path and image_path.exists():
            if zoom_level == 100:
                st.image(str(image_path), use_container_width=True)
            else:
                st.image(str(image_path), width=int(zoom_level * 8))
        else:
            st.error(f"Image for page {label} not found.")


def render_model_section(title, banner_class, record, attempts, filename, model_key, comments, reviewed, total):
    st.markdown(
        f'<div class="model-banner {banner_class}"><span>{html.escape(title)}</span>'
        f"<span>{reviewed} / {total} reviewed</span></div>",
        unsafe_allow_html=True,
    )
    if not record:
        st.warning("No record for this model.")
        return
    st.caption(
        f"Attempts: {len(attempts) if attempts else 1} · Status: {status_label(record)} · Questions: {len(qa_items(record))}"
    )
    if record.get("error"):
        st.error(record["error"])
    render_qa_cards(qa_items(record), filename, model_key, comments)
    with st.expander(f"🔍 {title} raw JSON", expanded=False):
        st.json(record)


def render_raw_json_inspector(records, filename):
    st.caption(f"Raw JSON outputs for `{filename}`")
    tabs = st.tabs([title for _, title, _ in MODELS])
    for tab, (model_key, title, _) in zip(tabs, MODELS):
        with tab:
            record = records.get(model_key)
            if record:
                st.download_button(
                    f"Download {title} JSON",
                    data=json.dumps(record, indent=2, ensure_ascii=False),
                    file_name=f"{model_key}_{filename}.json",
                    mime="application/json",
                    key=f"dl-{model_key}-{filename}",
                )
                st.json(record)
            else:
                st.info(f"No {title} record for this file.")


def reviewed_count(comments, name, model_key, record):
    entries = file_review_bucket(comments.get(name, {})).get(model_key, {})
    if not isinstance(entries, dict):
        return 0
    return sum(
        1
        for number in range(1, len(qa_items(record)) + 1)
        if isinstance(entries.get(str(number)), dict)
        and entries[str(number)].get("decision") in DECISIONS
    )


# Source selection (BI / OJK)
with st.sidebar:
    source_name = st.radio("Data Source", list(SOURCES), horizontal=True, key="source")
if st.session_state.get("_last_source") != source_name:
    st.session_state["_last_source"] = source_name
    st.session_state.record_index = 0
_source = SOURCES[source_name]
INTERNVL_JSONL = DATA_DIR / _source["internvl"]
QWEN_JSONL = DATA_DIR / _source["qwen"]
IMAGE_ROOTS = [BASE_DIR / name for name in _source["image_dirs"]]
IMAGE_ROOT = next((p for p in IMAGE_ROOTS if p.exists()), IMAGE_ROOTS[0])
SOURCES_IMAGES_NAME = _source["images_name"]
CLUSTER_PREFIX = f"{CLUSTER_ROOT}/{_source['images_name']}"

# Data loading
records_by_model = {
    "internvl": load_jsonl(str(INTERNVL_JSONL)),
    "qwen": load_jsonl(str(QWEN_JSONL)),
}
image_index = index_images(tuple(str(p) for p in IMAGE_ROOTS))

if not any(records_by_model.values()):
    st.error("Neither multi-page JSONL file (InternVL, Qwen) could be loaded.")
    st.stop()

by_file = {key: group_by_file(recs) for key, recs in records_by_model.items()}
all_files = sorted(set().union(*[set(d) for d in by_file.values()]))

if "record_index" not in st.session_state:
    st.session_state.record_index = 0

progress_slot = st.sidebar.container()

with st.sidebar:
    st.header("Find & Filter")
    search = st.text_input("Search page group", placeholder="e.g. p014 or Banten")
    review_filter = st.selectbox(
        "Review status",
        ["All reviews", "Unreviewed", "Retain", "Revise", "Discard"],
    )
    view_filter = st.selectbox(
        "Models present",
        [
            "All page groups",
            "In both models",
            "InternVL only",
            "Qwen only",
            "Has errors",
            "Missing image",
        ],
    )

    with st.expander("Display settings"):
        layout_split = st.selectbox(
            "Pane layout",
            [
                "55 / 45 (Image + Q&A)",
                "50 / 50 (Balanced)",
                "60 / 40 (Larger image)",
                "Stacked (Image top)",
            ],
            index=0,
        )
        sticky_image = st.checkbox(
            "Keep image in view while scrolling Q&A",
            value=True,
        )
        image_zoom = st.slider(
            "Image zoom",
            min_value=60,
            max_value=160,
            value=100,
            step=10,
            format="%d%%",
            help="100% fits the panel width. Zoom in to read small table cells; the image panel scrolls.",
        )

    with st.expander("Verification criteria"):
        st.markdown(
            """
1. **Query Clarity:** specific and unambiguous, targeting a particular topic.
2. **Answer Correctness:** factually correct and supported by the image. No hallucinations.
3. **Category Appropriateness:** the question matches its assigned category.
4. **Multi-Page Sources:** the answer needs more than one page, and every cited page (CITED PAGES) is correct.

**Decision rule:** Retain if all criteria are met. Revise for minor issues. Discard for major errors.
            """
        )

# Pin the image panel (and let it scroll on its own) next to the scrolling Q&A
if not layout_split.startswith("Stacked"):
    sticky_css = (
        "position: sticky; top: 3.4rem; align-self: flex-start;" if sticky_image else ""
    )
    st.markdown(
        f"""
        <style>
        div[data-testid="stColumn"]:has(.st-key-image-pane) {{ align-self: stretch !important; }}
        div[data-testid="stLayoutWrapper"]:has(> .st-key-image-pane) {{
            {sticky_css}
            width: 100%;
        }}
        .st-key-image-pane {{
            max-height: calc(100vh - 4.6rem);
            overflow: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

comments_index = load_comments()
file_stats = {}
filtered = []
for name in all_files:
    picked = {key: pick_record(by_file[key].get(name, [])) for key, _, _ in MODELS}
    reviewed = sum(reviewed_count(comments_index, name, key, picked[key]) for key, _, _ in MODELS)
    total = sum(len(qa_items(picked[key])) for key, _, _ in MODELS)
    file_stats[name] = (reviewed, total)

    present = {key for key, rec in picked.items() if rec}
    if view_filter == "In both models" and len(present) < 2:
        continue
    if view_filter == "InternVL only" and present != {"internvl"}:
        continue
    if view_filter == "Qwen only" and present != {"qwen"}:
        continue
    if view_filter == "Has errors" and not any(status_label(rec) == "error" for rec in picked.values() if rec):
        continue
    if view_filter == "Missing image" and all(
        path for _, path in resolve_images(next((r for r in picked.values() if r), None), image_index)
    ):
        continue
    if search and search.lower() not in name.lower():
        continue
    if review_filter == "Unreviewed":
        if total > 0 and reviewed >= total:
            continue
    elif review_filter != "All reviews":
        file_reviews = file_review_bucket(comments_index.get(name, {}))
        decisions = [
            entry.get("decision")
            for model_entry in file_reviews.values()
            if isinstance(model_entry, dict)
            for entry in model_entry.values()
            if isinstance(entry, dict)
        ]
        if review_filter not in decisions:
            continue
    filtered.append(name)

grand_reviewed = sum(r for r, _ in file_stats.values())
grand_total = sum(t for _, t in file_stats.values())
with progress_slot:
    st.markdown(f"**{source_name} review progress**")
    st.progress(grand_reviewed / grand_total if grand_total else 0.0)
    st.caption(f"{grand_reviewed} / {grand_total} questions reviewed")

if not filtered:
    st.warning("No records match the current filters.")
    st.stop()

st.session_state.record_index = max(
    0, min(st.session_state.record_index, len(filtered) - 1)
)


def previous_record():
    if st.session_state.record_index > 0:
        st.session_state.record_index -= 1


def next_record():
    if st.session_state.record_index < len(filtered) - 1:
        st.session_state.record_index += 1


def next_unreviewed():
    for position in range(st.session_state.record_index + 1, len(filtered)):
        reviewed, total = file_stats[filtered[position]]
        if reviewed < total:
            st.session_state.record_index = position
            return
    st.toast("No unreviewed files after this one.")


index = st.session_state.record_index
filename = filtered[index]
attempts = {key: by_file[key].get(filename, []) for key, _, _ in MODELS}
picked = {key: pick_record(attempts[key]) for key, _, _ in MODELS}
page_images = resolve_images(next((rec for rec in picked.values() if rec), None), image_index)
all_images_found = bool(page_images) and all(path for _, path in page_images)
group_pages = ", ".join(str(label) for label, _ in page_images)
file_reviewed, file_total = file_stats[filename]

with st.sidebar:
    st.divider()
    jump = st.number_input(
        f"Jump to record (1–{len(filtered)})",
        min_value=1,
        max_value=len(filtered),
        value=index + 1,
        step=1,
    )
    if jump != index + 1:
        st.session_state.record_index = jump - 1
        st.rerun()
    with st.expander("Data paths"):
        for key, title, _ in MODELS:
            st.caption(f"{title} JSONL")
            st.code(str({"internvl": INTERNVL_JSONL, "qwen": QWEN_JSONL}[key]))
        st.caption("Images root")
        st.code(str(IMAGE_ROOT))
        st.caption("Reviews stored in")
        st.code("Supabase" if supabase_config() else str(COMMENTS_PATH))

# Toolbar
bar_title, bar_prev, bar_next, bar_unrev = st.columns([5, 1, 1, 1.4], vertical_alignment="center")
with bar_title:
    st.markdown(
        f"""
        <div class="main-title">{html.escape(source_name)} · Group {index + 1} of {len(filtered)}
            &nbsp;<span style="font-size:13px;color:#6b7280;font-weight:600;">· {file_reviewed}/{file_total} reviewed</span></div>
        <div class="main-subtitle">📄 {html.escape(filename)} &nbsp;·&nbsp; pages {html.escape(group_pages)}</div>
        """,
        unsafe_allow_html=True,
    )
with bar_prev:
    st.button("← Prev", key="top_prev", on_click=previous_record, disabled=index == 0, use_container_width=True)
with bar_next:
    st.button("Next →", key="top_next", on_click=next_record, disabled=index >= len(filtered) - 1, use_container_width=True)
with bar_unrev:
    st.button("Next unreviewed ⏭", key="top_unrev", on_click=next_unreviewed, use_container_width=True)

if st.session_state.get("_save_error"):
    st.error(st.session_state["_save_error"])

# Image | Q&A
if layout_split.startswith("55"):
    left_pane, right_pane = st.columns([55, 45], gap="medium")
elif layout_split.startswith("50"):
    left_pane, right_pane = st.columns([1, 1], gap="medium")
elif layout_split.startswith("60"):
    left_pane, right_pane = st.columns([60, 40], gap="medium")
else:
    left_pane = st.container()
    right_pane = st.container()

with left_pane:
    with st.container(key="image-pane"):
        st.markdown(
            f"""
            <div class="image-panel-header">
                <span>🖼️ Source pages ({len(page_images)})</span>
                <span style="font-size:12px; font-weight:600; color:{'#16a34a' if all_images_found else '#dc2626'};">
                    {'● Images found' if all_images_found else '○ Image missing'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_source_images(page_images, zoom_level=image_zoom)

VIEW_OPTIONS = [title for _, title, _ in MODELS] + ["Compare all", "Raw JSON"]
with right_pane:
    default_view = next(
        (title for key, title, _ in MODELS if picked[key]),
        VIEW_OPTIONS[0],
    )
    view_mode = st.segmented_control(
        "View",
        VIEW_OPTIONS,
        default=default_view,
        key="view_mode",
        label_visibility="collapsed",
    ) or default_view

    comments = load_comments()

    def section(model_key, title, banner):
        record = picked[model_key]
        render_model_section(
            title,
            banner,
            record,
            attempts[model_key],
            filename,
            model_key,
            comments,
            reviewed_count(comments, filename, model_key, record),
            len(qa_items(record)),
        )

    if view_mode == "Compare all":
        for column, (model_key, title, banner) in zip(st.columns(2, gap="small"), MODELS):
            with column:
                section(model_key, title, banner)
    elif view_mode == "Raw JSON":
        render_raw_json_inspector(picked, filename)
    else:
        model_key, title, banner = next(m for m in MODELS if m[1] == view_mode)
        section(model_key, title, banner)

# Bottom navigation
st.divider()
bottom_left, bottom_center, bottom_unrev, bottom_right = st.columns([1, 2, 1.4, 1])
with bottom_left:
    st.button("← Previous", key="bottom_prev", on_click=previous_record, disabled=index == 0, use_container_width=True)
with bottom_center:
    st.markdown(
        f'<div class="record-counter">Record <b>{index + 1}</b> of <b>{len(filtered)}</b></div>',
        unsafe_allow_html=True,
    )
with bottom_unrev:
    st.button("Next unreviewed ⏭", key="bottom_unrev", on_click=next_unreviewed, use_container_width=True)
with bottom_right:
    st.button("Next →", key="bottom_next", on_click=next_record, disabled=index >= len(filtered) - 1, use_container_width=True)
