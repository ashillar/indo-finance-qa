import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "json"
DATA_ROOT = BASE_DIR
CLUSTER_ROOT = "/home/cluster-dgx1/ashillafryda/vllm-project"

SOURCES = {
    "BI": {
        "internvl": "bi_single_final_internvl.jsonl",
        "qwen": "bi_single_qwen25_vl.jsonl",
        "gemma": "bi_single_batch2_gemma3_4b.jsonl",
        "images": ["bi_images", "bi-single", "bi_single"],
    },
    "OJK": {
        "internvl": "ojk_single_final_internvl.jsonl",
        "qwen": "ojk_single_qwen25_vl.jsonl",
        "gemma": "ojk_single_final_gemma3_4b.jsonl",
        "images": ["ojk_images", "ojk-single", "ojk_single"],
    },
}

# Set from the sidebar source selector before any data is loaded.
INTERNVL_JSONL = QWEN_JSONL = GEMMA_JSONL = IMAGE_ROOT = None
CLUSTER_PREFIX = ""
COMMENTS_PATH = BASE_DIR / "qa_comments.json"

st.set_page_config(
    page_title="Model Evaluation: InternVL vs Qwen vs Gemma",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 98%;
    }
    .main-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 0;
        color: #111827;
    }
    .main-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-top: 3px;
        margin-bottom: 8px;
    }
    .model-banner {
        border-radius: 10px;
        padding: 9px 13px;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .banner-internvl {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .banner-qwen {
        background: #f5f3ff;
        color: #6d28d9;
        border: 1px solid #ddd6fe;
    }
    .banner-gemma {
        background: #f0fdfa;
        color: #0f766e;
        border: 1px solid #99f6e4;
    }
    .qa-card {
        border: 1px solid #e5e7eb;
        border-radius: 11px;
        padding: 13px 15px;
        margin-bottom: 12px;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .qa-number {
        font-size: 11px;
        font-weight: 700;
        color: #4f46e5;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 4px;
    }
    .qa-category {
        display: inline-block;
        font-size: 11px;
        color: #4b5563;
        background: #f3f4f6;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .qa-question {
        font-size: 13.5px;
        line-height: 1.5;
        font-weight: 600;
        color: #111827;
        margin-bottom: 8px;
    }
    .answer-label {
        font-size: 10px;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .answer-box {
        display: inline-block;
        padding: 7px 11px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 6px;
        color: #047857;
        font-size: 13px;
        font-weight: 700;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .answer-type {
        display: inline-block;
        margin-left: 6px;
        color: #9ca3af;
        font-size: 11px;
    }
    .image-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 8px;
        color: #1f2937;
    }
    .record-counter {
        text-align: center;
        font-size: 13px;
        color: #6b7280;
        padding-top: 6px;
    }
    section[data-testid="stSidebar"] {
        background: #f9fafb;
    }
    /* Smooth custom scrollbars for sticky containers */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
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
def index_images(root_string):
    root = Path(root_string)
    index = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".pdf"}:
            index.setdefault(path.name, []).append(str(path))
    return index


def remap_cluster_path(source_path):
    if not source_path:
        return None
    source = str(source_path)
    if source.startswith(CLUSTER_PREFIX):
        mapped = IMAGE_ROOT / source[len(CLUSTER_PREFIX):].lstrip("/")
        if mapped.exists():
            return mapped
    local = Path(source)
    if local.exists():
        return local
    return None


def resolve_image(record, image_index):
    if not record:
        return None
    mapped = remap_cluster_path(record.get("source_path"))
    if mapped:
        return mapped
    filename = os.path.basename(str(record.get("file", "")))
    hits = image_index.get(filename, [])
    if hits:
        return Path(hits[0])
    candidate = IMAGE_ROOT / filename
    if candidate.exists():
        return candidate
    return None


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
        "For multi-page queries, if the answer is derived from multiple pages, all referenced page sources must be accurately identified. Use N/A for single-page items.",
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
    return os.path.basename(str(record.get("file") or record.get("source_path") or ""))


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


def render_qa_cards(items, filename, model_key, comments):
    if not items:
        st.info("No Q&A parsed for this record.")
        return
    for number, qa in enumerate(items, start=1):
        if not isinstance(qa, dict):
            continue
        query = str(qa.get("query", ""))
        question = html.escape(query)
        answer = html.escape(str(qa.get("answer", "")))
        category = html.escape(str(qa.get("category", "Unknown")))
        answer_type = html.escape(str(qa.get("answer_type", "")))
        existing = saved_qa_entry(comments, filename, model_key, number)
        saved_criteria = existing.get("criteria") if isinstance(existing.get("criteria"), dict) else {}
        prefix = f"{filename}-{model_key}-{number}"

        st.markdown('<div class="qa-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="qa-number">Question {number}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="qa-category">{category}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="qa-question">{question}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="answer-label">ANSWER</div>',
            unsafe_allow_html=True,
        )
        if answer:
            st.markdown(
                f'<span class="answer-box">{answer}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No answer in this object.")
        if answer_type:
            st.markdown(
                f'<span class="answer-type">{answer_type}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="answer-label" style="margin-top:12px;">VERIFICATION CRITERIA</div>',
            unsafe_allow_html=True,
        )
        criterion_values = {}
        left_crit, right_crit = st.columns(2)
        for index, (field, label, help_text) in enumerate(CRITERIA):
            widget_key = f"crit-{prefix}-{field}"
            saved_value = saved_criteria.get(field, "Unreviewed")
            if saved_value not in CRITERION_CHOICES:
                saved_value = "Unreviewed"
            init_widget(widget_key, saved_value)
            target = left_crit if index % 2 == 0 else right_crit
            with target:
                criterion_values[field] = st.selectbox(
                    label,
                    CRITERION_CHOICES,
                    key=widget_key,
                    help=help_text,
                )

        decision_key = f"decision-{prefix}"
        saved_decision = existing.get("decision", "Unreviewed")
        if saved_decision not in DECISION_CHOICES:
            saved_decision = "Unreviewed"
        init_widget(decision_key, saved_decision)
        decision = st.radio(
            "Decision",
            DECISION_CHOICES,
            key=decision_key,
            horizontal=True,
            help="Retain if all criteria are met. Revise for minor issues (unclear query, wrong category). Discard for major errors that cannot be fixed reliably.",
        )

        st.markdown(
            '<div class="answer-label" style="margin-top:8px;">COMMENT</div>',
            unsafe_allow_html=True,
        )
        comment_key = f"comment-{prefix}"
        init_widget(comment_key, qa_comment(comments, filename, model_key, number))
        comment_value = st.text_area(
            "Comment",
            height=70,
            key=comment_key,
            placeholder="Optional notes: what to revise, why it is discarded, etc.",
            label_visibility="collapsed",
        )
        save_col, status_col = st.columns([1, 2])
        with save_col:
            saved = st.button(
                "Save review",
                key=f"save-{prefix}",
                use_container_width=True,
            )
        with status_col:
            if saved:
                save_qa_review(
                    filename,
                    model_key,
                    number,
                    {
                        "query": query,
                        "comment": comment_value.strip(),
                        "decision": decision,
                        "criteria": criterion_values,
                    },
                )
                st.success(f"Saved: {decision}")
            elif existing.get("updated_at"):
                st.caption(
                    f"{existing.get('decision', 'Unreviewed')} • {existing['updated_at']}"
                )

        st.markdown("</div>", unsafe_allow_html=True)


def render_source_image(image_path, record, zoom_level=100):
    if image_path and image_path.exists():
        if image_path.suffix.lower() == ".pdf":
            st.caption(f"📄 PDF Document: {image_path.name}")
            st.download_button(
                "Download PDF",
                data=image_path.read_bytes(),
                file_name=image_path.name,
                mime="application/pdf",
                key=f"pdf-{image_path.name}",
                use_container_width=True,
            )
        else:
            if zoom_level == 100:
                st.image(str(image_path), use_container_width=True)
            else:
                st.image(str(image_path), width=int(zoom_level * 7))
        st.caption(f"📁 `{image_path}`")
        return
    st.error(f"Source image not found under {IMAGE_ROOT.name}.")
    if record:
        st.code(str(record.get("source_path") or record.get("file") or ""))


def render_model_section(title, banner_class, record, attempts, filename, model_key, comments):
    st.markdown(
        f'<div class="model-banner {banner_class}"><span>{html.escape(title)}</span><span>QA: {len(qa_items(record))}</span></div>',
        unsafe_allow_html=True,
    )

    if not record:
        st.warning("No record for this model.")
        return

    cols = st.columns(3)
    cols[0].metric("Attempts", len(attempts) if attempts else 1)
    cols[1].metric("Status", status_label(record))
    cols[2].metric("QA count", len(qa_items(record)))

    if record.get("error"):
        st.error(record["error"])

    st.markdown("##### Questions & Answers")
    render_qa_cards(qa_items(record), filename, model_key, comments)

    with st.expander(f"🔍 View {title} Raw JSON", expanded=False):
        st.json(record)


def render_raw_json_inspector(internvl_record, qwen_record, gemma_record, filename):
    st.markdown("### 📋 Raw JSON Inspector")
    st.caption(f"View and inspect raw JSON outputs for file: `{filename}`")
    tab_gemma, tab_inter, tab_qwen = st.tabs(["🟢 Gemma 3", "🔵 InternVL", "🟣 Qwen2.5-VL"])

    with tab_gemma:
        if gemma_record:
            st.download_button(
                "Download Gemma JSON",
                data=json.dumps(gemma_record, indent=2, ensure_ascii=False),
                file_name=f"gemma_{filename}.json",
                mime="application/json",
                key=f"dl-gemma-{filename}",
            )
            st.json(gemma_record)
        else:
            st.info("No Gemma record available for this file.")

    with tab_inter:
        if internvl_record:
            st.download_button(
                "Download InternVL JSON",
                data=json.dumps(internvl_record, indent=2, ensure_ascii=False),
                file_name=f"internvl_{filename}.json",
                mime="application/json",
                key=f"dl-internvl-{filename}",
            )
            st.json(internvl_record)
        else:
            st.info("No InternVL record available for this file.")

    with tab_qwen:
        if qwen_record:
            st.download_button(
                "Download Qwen JSON",
                data=json.dumps(qwen_record, indent=2, ensure_ascii=False),
                file_name=f"qwen_{filename}.json",
                mime="application/json",
                key=f"dl-qwen-{filename}",
            )
            st.json(qwen_record)
        else:
            st.info("No Qwen record available for this file.")


# Source selection (BI / OJK)
with st.sidebar:
    source_name = st.radio("Data Source", list(SOURCES), horizontal=True, key="source")
if st.session_state.get("_last_source") != source_name:
    st.session_state["_last_source"] = source_name
    st.session_state.record_index = 0
_source = SOURCES[source_name]
INTERNVL_JSONL = DATA_DIR / _source["internvl"]
QWEN_JSONL = DATA_DIR / _source["qwen"]
GEMMA_JSONL = DATA_DIR / _source["gemma"]
IMAGE_ROOT = next(
    (DATA_ROOT / name for name in _source["images"] if (DATA_ROOT / name).exists()),
    DATA_ROOT / _source["images"][0],
)
CLUSTER_PREFIX = f"{CLUSTER_ROOT}/{_source['images'][0]}"

# Data loading
internvl_records = load_jsonl(str(INTERNVL_JSONL))
qwen_records = load_jsonl(str(QWEN_JSONL))
gemma_records = load_jsonl(str(GEMMA_JSONL))

image_index = index_images(str(IMAGE_ROOT))

if not internvl_records and not qwen_records and not gemma_records:
    st.error("None of the JSONL files (InternVL, Qwen, Gemma) could be loaded.")
    st.stop()

internvl_by_file = group_by_file(internvl_records)
qwen_by_file = group_by_file(qwen_records)
gemma_by_file = group_by_file(gemma_records)
all_files = sorted(set(internvl_by_file) | set(qwen_by_file) | set(gemma_by_file))

if "record_index" not in st.session_state:
    st.session_state.record_index = 0

with st.sidebar:
    st.header("Navigation & Filter")
    st.caption(f"{source_name} Single-Page Model Comparison")

    view_filter = st.selectbox(
        "Model Filter",
        [
            "All files",
            "In all 3 models",
            "InternVL only",
            "Qwen only",
            "Gemma only",
            "Has errors",
            "Missing image",
        ],
    )
    search = st.text_input("Search filename", placeholder="e.g. p013.png")

    review_filter = st.selectbox(
        "Review Status",
        [
            "All reviews",
            "Unreviewed",
            "Retain",
            "Revise",
            "Discard",
        ],
    )

    st.divider()
    st.subheader("Layout Settings")
    layout_split = st.selectbox(
        "Pane Layout",
        [
            "50 / 50 (Balanced Side-by-Side)",
            "45 / 55 (More Models Space)",
            "60 / 40 (Larger Image)",
            "Stacked (Image Top, Models Below)",
        ],
        index=0,
        help="Side-by-side lets you inspect images and model JSON/questions simultaneously.",
    )
    sticky_image = st.checkbox(
        "Pin Image Panel (Sticky)",
        value=True,
        help="Keeps the source image in view as you scroll through questions on the right.",
    )
    image_zoom = st.slider(
        "Image Zoom",
        min_value=60,
        max_value=160,
        value=100,
        step=10,
        format="%d%%",
        help="Zoom in to see small table cells or chart labels clearly.",
    )

    with st.expander("Verification Criteria"):
        st.markdown(
            """
1. **Query Clarity:** The query should be specific and unambiguous, targeting a particular topic in a document.
2. **Answer Correctness:** The answer must be factually correct and directly supported by the visual content. No hallucinations.
3. **Category Appropriateness:** The question should match its assigned category.
4. **Multi-Page Sources:** All referenced page sources must be accurately identified. Use N/A for single-page.

**Decision rule:** Retain if all criteria are met. Revise for minor issues. Discard for major errors.
            """
        )

# Inject sticky column CSS when in side-by-side mode and sticky is enabled
if sticky_image and not layout_split.startswith("Stacked"):
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
            position: sticky !important;
            top: 1rem !important;
            align-self: flex-start !important;
            max-height: calc(100vh - 2rem) !important;
            overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

filtered = []
comments_index = load_comments()
for name in all_files:
    internvl = pick_record(internvl_by_file.get(name, []))
    qwen = pick_record(qwen_by_file.get(name, []))
    gemma = pick_record(gemma_by_file.get(name, []))

    image = resolve_image(internvl or qwen or gemma, image_index)
    internvl_status = status_label(internvl)
    qwen_status = status_label(qwen)
    gemma_status = status_label(gemma)

    if view_filter == "In all 3 models" and not (internvl and qwen and gemma):
        continue
    if view_filter == "InternVL only" and not (internvl and not qwen and not gemma):
        continue
    if view_filter == "Qwen only" and not (qwen and not internvl and not gemma):
        continue
    if view_filter == "Gemma only" and not (gemma and not internvl and not qwen):
        continue
    if (
        view_filter == "Has errors"
        and internvl_status != "error"
        and qwen_status != "error"
        and gemma_status != "error"
    ):
        continue
    if view_filter == "Missing image" and image is not None:
        continue
    if search and search.lower() not in name.lower():
        continue
    if review_filter != "All reviews":
        file_reviews = file_review_bucket(comments_index.get(name, {}))
        decisions = []
        for model_key in ("internvl", "qwen", "gemma"):
            model_entry = file_reviews.get(model_key, {})
            if not isinstance(model_entry, dict):
                continue
            for entry in model_entry.values():
                if isinstance(entry, dict):
                    decisions.append(entry.get("decision", "Unreviewed"))
        if review_filter == "Unreviewed":
            internvl_n = len(qa_items(internvl))
            qwen_n = len(qa_items(qwen))
            gemma_n = len(qa_items(gemma))
            total_qa = internvl_n + qwen_n + gemma_n
            reviewed = sum(1 for d in decisions if d and d != "Unreviewed")
            if reviewed >= total_qa and total_qa > 0:
                continue
        elif review_filter not in decisions:
            continue
    filtered.append(name)

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


index = st.session_state.record_index
filename = filtered[index]

internvl_attempts = internvl_by_file.get(filename, [])
qwen_attempts = qwen_by_file.get(filename, [])
gemma_attempts = gemma_by_file.get(filename, [])

internvl_record = pick_record(internvl_attempts)
qwen_record = pick_record(qwen_attempts)
gemma_record = pick_record(gemma_attempts)

image_path = (
    resolve_image(internvl_record, image_index)
    or resolve_image(qwen_record, image_index)
    or resolve_image(gemma_record, image_index)
)

# Header & Top Navigation Toolbar
top_meta, top_prev_col, top_next_col = st.columns([3, 1, 1])
with top_meta:
    st.markdown(
        f"""
        <div class="main-title">InternVL vs Qwen2.5-VL vs Gemma 3</div>
        <div class="main-subtitle">
            📄 <b>{html.escape(filename)}</b>
            &nbsp;•&nbsp;
            Record <b>{index + 1}</b> of <b>{len(filtered)}</b>
            &nbsp;•&nbsp;
            🔵 InternVL: {len(internvl_records)} &nbsp;|&nbsp; 🟣 Qwen: {len(qwen_records)} &nbsp;|&nbsp; 🟢 Gemma: {len(gemma_records)}
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_prev_col:
    st.button(
        "← Prev File",
        key="top_prev",
        on_click=previous_record,
        disabled=index == 0,
        use_container_width=True,
    )
with top_next_col:
    st.button(
        "Next File →",
        key="top_next",
        on_click=next_record,
        disabled=index >= len(filtered) - 1,
        use_container_width=True,
    )

with st.sidebar:
    st.divider()
    st.metric("Current File Index", f"{index + 1} / {len(filtered)}")
    jump = st.number_input(
        "Jump to record",
        min_value=1,
        max_value=len(filtered),
        value=index + 1,
        step=1,
    )
    if jump != index + 1:
        st.session_state.record_index = jump - 1
        st.rerun()

    st.divider()
    st.caption(f"Active: {filename}")
    if image_path:
        st.success("Image found locally")
    else:
        st.error("Image missing locally")
    with st.expander("JSONL & Data Paths"):
        st.caption("InternVL JSONL")
        st.code(str(INTERNVL_JSONL))
        st.caption("Qwen2.5-VL JSONL")
        st.code(str(QWEN_JSONL))
        st.caption("Gemma 3 JSONL")
        st.code(str(GEMMA_JSONL))
        st.caption("Images Root")
        st.code(str(IMAGE_ROOT))
        st.caption("Reviews stored in")
        st.code("Supabase" if supabase_config() else str(COMMENTS_PATH))

# Side-by-Side Split or Stacked Layout
if layout_split.startswith("50 / 50"):
    left_pane, right_pane = st.columns([1, 1], gap="medium")
elif layout_split.startswith("45 / 55"):
    left_pane, right_pane = st.columns([45, 55], gap="medium")
elif layout_split.startswith("60 / 40"):
    left_pane, right_pane = st.columns([60, 40], gap="medium")
else:
    # Stacked
    left_pane = st.container()
    right_pane = st.container()

with left_pane:
    st.markdown(
        f"""
        <div class="image-panel-header">
            <span>🖼️ Source Document</span>
            <span style="font-size:12px; font-weight:600; color:{'#16a34a' if image_path else '#dc2626'};">
                {'● Image Available' if image_path else '○ Image Missing'}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_source_image(
        image_path,
        internvl_record or qwen_record or gemma_record,
        zoom_level=image_zoom,
    )

with right_pane:
    # Mode Pills to switch between 3-model side-by-side or focused model views or raw JSON
    view_mode = st.pills(
        "View Mode",
        [
            "🌟 All 3 Models",
            "🟢 Gemma 3",
            "🔵 InternVL",
            "🟣 Qwen2.5-VL",
            "📋 Raw JSON Inspector",
        ],
        default="🌟 All 3 Models",
        label_visibility="collapsed",
    )

    comments = load_comments()

    if view_mode == "🌟 All 3 Models":
        col_gemma, col_internvl, col_qwen = st.columns(3, gap="small")
        with col_gemma:
            render_model_section(
                "Gemma 3",
                "banner-gemma",
                gemma_record,
                gemma_attempts,
                filename,
                "gemma",
                comments,
            )
        with col_internvl:
            render_model_section(
                "InternVL",
                "banner-internvl",
                internvl_record,
                internvl_attempts,
                filename,
                "internvl",
                comments,
            )
        with col_qwen:
            render_model_section(
                "Qwen2.5-VL",
                "banner-qwen",
                qwen_record,
                qwen_attempts,
                filename,
                "qwen",
                comments,
            )
    elif view_mode == "🟢 Gemma 3":
        render_model_section(
            "Gemma 3",
            "banner-gemma",
            gemma_record,
            gemma_attempts,
            filename,
            "gemma",
            comments,
        )
    elif view_mode == "🔵 InternVL":
        render_model_section(
            "InternVL",
            "banner-internvl",
            internvl_record,
            internvl_attempts,
            filename,
            "internvl",
            comments,
        )
    elif view_mode == "🟣 Qwen2.5-VL":
        render_model_section(
            "Qwen2.5-VL",
            "banner-qwen",
            qwen_record,
            qwen_attempts,
            filename,
            "qwen",
            comments,
        )
    elif view_mode == "📋 Raw JSON Inspector":
        render_raw_json_inspector(
            internvl_record, qwen_record, gemma_record, filename
        )

# Bottom Navigation
st.divider()
bottom_left, bottom_center, bottom_right = st.columns([1, 2, 1])
with bottom_left:
    st.button(
        "← Previous",
        key="bottom_prev",
        on_click=previous_record,
        disabled=index == 0,
        use_container_width=True,
    )
with bottom_center:
    st.markdown(
        f'<div class="record-counter">Record <b>{index + 1}</b> of <b>{len(filtered)}</b> ({html.escape(filename)})</div>',
        unsafe_allow_html=True,
    )
with bottom_right:
    st.button(
        "Next →",
        key="bottom_next",
        on_click=next_record,
        disabled=index >= len(filtered) - 1,
        use_container_width=True,
    )
