import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / os.environ.get("QA_DATA", "bi_multipage_qwen25_vl.jsonl")
IMAGE_DIRS = [BASE_DIR / "bi_multi", BASE_DIR / "bi_images" / "multi", BASE_DIR / "bi_images"]
FEEDBACK_PATH = DATA_PATH.with_name(f"{DATA_PATH.stem}_feedback.json")

CRITERIA = {
    "query_clarity": (
        "Query Clarity",
        "Specific and unambiguous, targeting a particular topic.",
    ),
    "answer_correctness": (
        "Answer Correctness",
        "Factually correct and supported by the image. No hallucinations.",
    ),
    "category_appropriateness": (
        "Category Appropriateness",
        "The question matches its assigned category.",
    ),
    "multipage_sources": (
        "Multi-Page Sources",
        "All referenced page sources are identified. Use N/A for single-page.",
    ),
}
PASS_OPTIONS = ["Yes", "No"]
MULTIPAGE_OPTIONS = ["Yes", "No", "N/A"]
DECISIONS = ["Retain", "Revise", "Discard"]

st.set_page_config(page_title="Multi-Page QA Annotation", page_icon="📑", layout="wide")


def salvage_questions(raw_output):
    """Recover questions from a failed generation's raw output, if it is valid JSON."""
    try:
        parsed = json.loads(raw_output or "")
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get("result"), list):
        items = parsed["result"]
    elif isinstance(parsed, dict):
        items = [v for k, v in sorted(parsed.items()) if isinstance(v, dict)]
    else:
        items = parsed if isinstance(parsed, list) else []
    return [q for q in items if isinstance(q, dict) and "query" in q]


@st.cache_data
def load_records(path_string):
    records = []
    with open(path_string, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("qa"):
                record["questions"] = record["qa"].get("result", [])
                record["failed"] = False
            else:
                record["questions"] = salvage_questions(record.get("raw_output"))
                record["failed"] = True
            records.append(record)
    return records


def resolve_image(source_path):
    name = os.path.basename(source_path)
    for directory in IMAGE_DIRS:
        candidate = directory / name
        if candidate.exists():
            return candidate
    candidate = BASE_DIR / source_path
    return candidate if candidate.exists() else None


def load_feedback():
    if FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_feedback(feedback):
    fd, tmp_path = tempfile.mkstemp(dir=BASE_DIR, suffix=".json.tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, FEEDBACK_PATH)


def is_complete(entry):
    if not entry or not entry.get("decision"):
        return False
    return all(entry.get("criteria", {}).get(k, {}).get("verdict") for k in CRITERIA)


def group_status(record, feedback):
    saved = feedback.get(record["group_key"], {}).get("questions", {})
    total = len(record["questions"])
    done = sum(is_complete(saved.get(str(i))) for i in range(total))
    return done, total


def option_index(options, value):
    return options.index(value) if value in options else None


feedback = load_feedback()
records = load_records(str(DATA_PATH))

with st.sidebar:
    st.title("📑 Multi-Page QA Review")
    annotator = st.text_input("Annotator name", value=st.session_state.get("annotator", ""))
    st.session_state["annotator"] = annotator

    include_failed = st.checkbox(
        "Include failed generations",
        value=False,
        help="Records whose QA failed validation; questions are recovered from raw_output.",
    )
    status_filter = st.radio("Show", ["All", "Not finished", "Finished"], horizontal=True)

    visible = [r for r in records if include_failed or not r["failed"]]
    if status_filter != "All":
        want_done = status_filter == "Finished"
        visible = [
            r for r in visible
            if (lambda d, t: (t > 0 and d == t) == want_done)(*group_status(r, feedback))
        ]

    if not visible:
        st.warning("No records match the current filter.")
        st.stop()

    def label(i):
        r = visible[i]
        done, total = group_status(r, feedback)
        mark = "✅" if total and done == total else ("🟡" if done else "⬜")
        flag = " ⚠️" if r["failed"] else ""
        return f"{mark} {i + 1}. {r['group_key']} ({done}/{total}){flag}"

    if st.session_state.get("group_idx", 0) >= len(visible):
        st.session_state["group_idx"] = 0

    def step(delta):
        st.session_state["group_idx"] = (st.session_state["group_idx"] + delta) % len(visible)

    st.selectbox("Group", range(len(visible)), format_func=label, key="group_idx")
    nav_prev, nav_next = st.columns(2)
    nav_prev.button("◀ Prev", on_click=step, args=(-1,), width="stretch")
    nav_next.button("Next ▶", on_click=step, args=(1,), width="stretch")

    all_done = sum(group_status(r, feedback)[0] for r in records if include_failed or not r["failed"])
    all_total = sum(len(r["questions"]) for r in records if include_failed or not r["failed"])
    st.progress(all_done / all_total if all_total else 0.0, text=f"{all_done}/{all_total} questions annotated")

    image_share = st.slider("Image column width (%)", 30, 75, 55, step=5)
    st.download_button(
        "⬇️ Download feedback JSON",
        data=json.dumps(feedback, ensure_ascii=False, indent=2),
        file_name=FEEDBACK_PATH.name,
        mime="application/json",
        width="stretch",
    )
    st.caption(f"Saved to `{FEEDBACK_PATH.name}`")

record = visible[st.session_state["group_idx"]]
group_key = record["group_key"]
saved_group = feedback.get(group_key, {}).get("questions", {})

st.header(record["document"])
st.caption(f"Group: `{group_key}` · Pages: {', '.join(map(str, record['pages']))} · Attempts: {record.get('attempts')}")
if record["failed"]:
    st.error(f"Generation failed validation: {record.get('error')}")
if record.get("near_duplicate_queries"):
    with st.expander("⚠️ Near-duplicate queries flagged by the generator"):
        for q in record["near_duplicate_queries"]:
            st.markdown(f"- {q}")


def render_pages(qi, source_paths, pages):
    paths = [resolve_image(p) for p in source_paths]
    tabs = st.tabs([f"Page {p}" for p in pages] + (["Side by side"] if len(paths) > 1 else []))
    for tab, page, path in zip(tabs, pages, paths):
        with tab:
            if path:
                st.image(str(path), width="stretch")
            else:
                st.error(f"Image for page {page} not found.")
    if len(paths) > 1:
        with tabs[-1]:
            for col, page, path in zip(st.columns(len(paths)), pages, paths):
                if path:
                    col.image(str(path), caption=f"Page {page}", width="stretch")


def render_feedback_form(qi, q, prev, source_pages, single_page):
    prev_criteria = prev.get("criteria", {})
    with st.form(key=f"form::{group_key}::{qi}"):
        answers = {}
        for ck, (title, desc) in CRITERIA.items():
            options = MULTIPAGE_OPTIONS if ck == "multipage_sources" else PASS_OPTIONS
            prev_verdict = prev_criteria.get(ck, {}).get("verdict")
            if prev_verdict is None and ck == "multipage_sources" and single_page:
                prev_verdict = "N/A"
            c1, c2 = st.columns([1, 1])
            verdict = c1.radio(
                f"**{title}**",
                options,
                index=option_index(options, prev_verdict),
                help=desc,
                horizontal=True,
                key=f"{group_key}::{qi}::{ck}::verdict",
            )
            comment = c2.text_input(
                f"{title} comment",
                value=prev_criteria.get(ck, {}).get("comment", ""),
                placeholder=desc,
                key=f"{group_key}::{qi}::{ck}::comment",
            )
            answers[ck] = {"verdict": verdict, "comment": comment}

        decision = st.radio(
            "**Decision**",
            DECISIONS,
            index=option_index(DECISIONS, prev.get("decision")),
            horizontal=True,
            key=f"{group_key}::{qi}::decision",
        )
        revised_query = st.text_area(
            "Revised query (if Revise)",
            value=prev.get("revised_query", ""),
            key=f"{group_key}::{qi}::revised_query",
        )
        revised_answer = st.text_area(
            "Revised answer (if Revise)",
            value=prev.get("revised_answer", ""),
            key=f"{group_key}::{qi}::revised_answer",
        )
        general_comment = st.text_area(
            "General comment",
            value=prev.get("comment", ""),
            key=f"{group_key}::{qi}::comment",
        )
        submitted = st.form_submit_button("💾 Save feedback", type="primary")

    if not submitted:
        return
    missing = [CRITERIA[k][0] for k, v in answers.items() if v["verdict"] is None]
    if decision is None:
        missing.append("Decision")
    if missing:
        st.warning("Not saved yet — please fill: " + ", ".join(missing))
        return

    feedback = load_feedback()
    group_entry = feedback.setdefault(group_key, {
        "document": record["document"],
        "pages": record["pages"],
        "source_paths": record["source_paths"],
        "generation_failed": record["failed"],
        "questions": {},
    })
    group_entry["questions"][str(qi)] = {
        "query": q.get("query"),
        "answer": q.get("answer"),
        "category": q.get("category"),
        "answer_type": q.get("answer_type"),
        "source_pages": source_pages,
        "criteria": answers,
        "decision": decision,
        "revised_query": revised_query,
        "revised_answer": revised_answer,
        "comment": general_comment,
        "annotator": annotator,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_feedback(feedback)
    st.toast(f"Saved Q{qi + 1} of {group_key}", icon="✅")
    st.rerun()


if not record["questions"]:
    st.info("No questions available for this group.")
    render_pages("none", record["source_paths"], record["pages"])
    if record.get("raw_output"):
        with st.expander("Raw output"):
            st.code(record["raw_output"])

for qi, q in enumerate(record["questions"]):
    prev = saved_group.get(str(qi), {})
    source_pages = q.get("source_pages") or []
    single_page = len(set(source_pages)) <= 1
    q_paths = q.get("source_path") or record["source_paths"]
    if isinstance(q_paths, str):
        q_paths = [q_paths]
    q_pages = source_pages if len(source_pages) == len(q_paths) else record["pages"][: len(q_paths)]
    status = "✅" if is_complete(prev) else "⬜"

    with st.container(border=True):
        st.markdown(f"#### {status} Q{qi + 1} · `{q.get('category', '-')}` · {q.get('answer_type', '-')}")
        image_col, qa_col = st.columns([image_share, 100 - image_share], gap="large")
        with image_col:
            render_pages(qi, q_paths, q_pages)
        with qa_col:
            st.markdown(f"**Query:** {q.get('query', '')}")
            st.markdown(f"**Answer:** {q.get('answer', '')}")
            st.caption(f"Source pages: {', '.join(map(str, source_pages)) or '-'}"
                       + (" (single-page → use N/A for Multi-Page Sources)" if single_page else ""))
            render_feedback_form(qi, q, prev, source_pages, single_page)

