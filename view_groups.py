import html
import json
import os
from pathlib import Path
import re
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Multi-Page Group Viewer (BI & OJK)",
    page_icon="🖼️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

SOURCES = {
    "Bank Indonesia (BI)": {
        "jsonl": BASE_DIR / "json" / "bi_multipage_qwen_v4_new.jsonl",
        "dirs": [BASE_DIR / "bi_multi", BASE_DIR / "bi_images"],
    },
    "Otoritas Jasa Keuangan (OJK)": {
        "jsonl": BASE_DIR / "json" / "ojk_multipage_qwen25_vl_v4.jsonl",
        "dirs": [BASE_DIR / "ojk_multi", BASE_DIR / "ojk_images"],
    },
}

def highlight_html(text: str, query: str) -> str:
    """Safely escapes text and highlights query matches."""
    if not text:
        return ""
    escaped = html.escape(str(text))
    if not query:
        return escaped
    pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
    return pattern.sub(
        lambda m: f"<mark style='background-color: #ffe066; color: #1a1a1a; padding: 2px 4px; border-radius: 3px; font-weight: 600;'>{m.group(0)}</mark>",
        escaped,
    )

@st.cache_data
def build_image_index():
    index = {}
    for folder_name in ["bi_multi", "ojk_multi", "bi_images", "ojk_images"]:
        folder = BASE_DIR / folder_name
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                index[p.name] = p
    return index

image_index = build_image_index()

@st.cache_data
def load_groups(source_key):
    config = SOURCES[source_key]
    jsonl_path = config["jsonl"]
    groups = []
    
    if jsonl_path.exists():
        seen_keys = set()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                key = data.get("group_key")
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    doc = data.get("document", "Unknown")
                    pages = data.get("pages", [])
                    source_paths = data.get("source_paths", [])
                    
                    resolved_images = []
                    for i, sp in enumerate(source_paths):
                        fname = Path(sp).name
                        local_path = image_index.get(fname)
                        page_num = pages[i] if i < len(pages) else i + 1
                        resolved_images.append({
                            "page": page_num,
                            "filename": fname,
                            "path": local_path,
                        })

                    # Parse QA and extracted evidence text
                    qa_data = data.get("qa", {})
                    qa_results = []
                    if isinstance(qa_data, dict):
                        qa_results = qa_data.get("result", [])

                    searchable_parts = [doc, key]
                    for sp in source_paths:
                        searchable_parts.append(Path(sp).name)

                    # Map page number to evidence statements extracted from that page
                    page_evidence_map = {}
                    for qa_item in qa_results:
                        q = qa_item.get("query", "")
                        a = qa_item.get("answer", "")
                        if q:
                            searchable_parts.append(q)
                        if a:
                            searchable_parts.append(a)
                        for ev in qa_item.get("evidence", []):
                            ev_page = ev.get("page")
                            stmt = ev.get("statement", "")
                            subj = ev.get("subject", "")
                            val = str(ev.get("value", "") or "")
                            period = str(ev.get("period", "") or "")
                            for text_item in [stmt, subj, val, period]:
                                if text_item:
                                    searchable_parts.append(text_item)
                            if ev_page is not None and stmt:
                                page_evidence_map.setdefault(ev_page, []).append({
                                    "statement": stmt,
                                    "subject": subj,
                                    "type": ev.get("type", "evidence"),
                                })

                    searchable_text = " ".join(searchable_parts).lower()
                    
                    groups.append({
                        "key": key,
                        "document": doc,
                        "pages": pages,
                        "images": resolved_images,
                        "qa_results": qa_results,
                        "page_evidence": page_evidence_map,
                        "searchable_text": searchable_text,
                    })
    return groups

# --- Sidebar UI ---
st.sidebar.title("🖼️ Group Image Viewer")
selected_source = st.sidebar.radio("Select Institution:", list(SOURCES.keys()))

# Reset index if source changed
if "current_source" not in st.session_state:
    st.session_state.current_source = selected_source
elif st.session_state.current_source != selected_source:
    st.session_state.current_source = selected_source
    st.session_state.group_idx = 0
    st.session_state.nav_select = 0

all_groups = load_groups(selected_source)

if not all_groups:
    st.error(f"No groups found for {selected_source}.")
    st.stop()

# --- Search in Groups/Images ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Search in Images")
search_query = st.sidebar.text_input(
    "Search word or phrase:",
    value="",
    placeholder="e.g. inflasi, kredit, karbon, Bali",
    help="Searches document titles, questions, answers, and extracted page evidence (statements, tables, charts)."
).strip()

# Reset index if search query changed
if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = search_query
elif st.session_state.last_search_query != search_query:
    st.session_state.last_search_query = search_query
    st.session_state.group_idx = 0
    st.session_state.nav_select = 0

# Filter groups based on search query
if search_query:
    q_lower = search_query.lower()
    groups = [g for g in all_groups if q_lower in g["searchable_text"]]
    st.sidebar.caption(f"Found **{len(groups)}** of {len(all_groups)} groups matching `\"{search_query}\"`")
else:
    groups = all_groups

if not groups:
    st.warning(f"No groups found containing **'{search_query}'**.")
    st.info("Try another word or clear the search box to view all groups.")
    st.stop()

# Initialize session state for group navigation
if "group_idx" not in st.session_state:
    st.session_state.group_idx = 0
    st.session_state.nav_select = 0

# Ensure index within bounds
if st.session_state.group_idx >= len(groups):
    st.session_state.group_idx = 0
    st.session_state.nav_select = 0

# Navigation controls in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")

def prev_group():
    st.session_state.group_idx = (st.session_state.group_idx - 1) % len(groups)
    st.session_state.nav_select = st.session_state.group_idx

def next_group():
    st.session_state.group_idx = (st.session_state.group_idx + 1) % len(groups)
    st.session_state.nav_select = st.session_state.group_idx

def jump_group():
    st.session_state.group_idx = st.session_state.nav_select

col_prev, col_next = st.sidebar.columns(2)
with col_prev:
    st.button("⬅️ Previous", on_click=prev_group, use_container_width=True)
with col_next:
    st.button("Next ➡️", on_click=next_group, use_container_width=True)

group_options = [
    f"{i+1}. {g['document']} (Pages: {', '.join(map(str, g['pages']))})"
    for i, g in enumerate(groups)
]

st.sidebar.selectbox(
    "Jump to Group:",
    options=range(len(groups)),
    format_func=lambda i: group_options[i],
    index=st.session_state.group_idx,
    key="nav_select",
    on_change=jump_group,
)

# View settings
st.sidebar.markdown("---")
st.sidebar.subheader("Display Settings")
image_width_mode = st.sidebar.radio(
    "Image Sizing:",
    ["Fit to Column (Side-by-Side)", "Custom Zoom"],
    index=0
)
zoom_width = 500
if image_width_mode == "Custom Zoom":
    zoom_width = st.sidebar.slider("Image Width (px):", min_value=250, max_value=1200, value=500, step=50)

# --- Main Content ---
current_group = groups[st.session_state.group_idx]
num_images = len(current_group["images"])

# Header
doc_title = current_group['document']
if search_query:
    st.markdown(f"## {highlight_html(doc_title, search_query)}", unsafe_allow_html=True)
else:
    st.subheader(doc_title)

status_filter = f" *(filtered by '{search_query}')*" if search_query else ""
st.caption(
    f"Group **{st.session_state.group_idx + 1} of {len(groups)}**{status_filter} | "
    f"Group Key: `{current_group['key']}` | "
    f"Pages: **{', '.join(str(p) for p in current_group['pages'])}**"
)

# Search Match Summary Banner
if search_query:
    matched_evidences = []
    for p_num, ev_list in current_group.get("page_evidence", {}).items():
        for ev in ev_list:
            if search_query.lower() in ev["statement"].lower() or search_query.lower() in ev.get("subject", "").lower():
                matched_evidences.append((p_num, ev))

    matched_qa = []
    for qa_item in current_group.get("qa_results", []):
        q = qa_item.get("query", "")
        a = qa_item.get("answer", "")
        if search_query.lower() in q.lower() or search_query.lower() in a.lower():
            matched_qa.append(qa_item)

    if matched_evidences or matched_qa:
        with st.container():
            st.markdown(
                f"""<div style="background-color: #f0f7ff; border-left: 4px solid #1976d2; padding: 10px 14px; border-radius: 4px; margin-bottom: 12px;">
                <b style="color: #0d47a1;">🔍 Found matches in this group for:</b> <code>{html.escape(search_query)}</code>
                </div>""",
                unsafe_allow_html=True,
            )

# Render images side-by-side using Streamlit columns
if num_images > 0:
    cols = st.columns(num_images)
    for col, img_info in zip(cols, current_group["images"]):
        with col:
            st.markdown(f"### Page {img_info['page']}")
            st.caption(f"`{img_info['filename']}`")

            # Check if this specific page has extracted evidence matching the search query
            if search_query:
                page_evs = current_group.get("page_evidence", {}).get(img_info["page"], [])
                matching_page_evs = [
                    ev for ev in page_evs
                    if search_query.lower() in ev["statement"].lower() or search_query.lower() in ev.get("subject", "").lower()
                ]
                if matching_page_evs:
                    for ev in matching_page_evs:
                        highlighted_stmt = highlight_html(ev["statement"], search_query)
                        subj_html = f"<b>[{html.escape(ev['subject'])}]</b> " if ev.get("subject") else ""
                        st.markdown(
                            f"<div style='background-color: #fffde7; border: 1px solid #ffd54f; padding: 6px 10px; border-radius: 4px; font-size: 0.88em; margin-bottom: 8px;'>"
                            f"📌 {subj_html}{highlighted_stmt}</div>",
                            unsafe_allow_html=True,
                        )

            if img_info["path"] and img_info["path"].exists():
                try:
                    img = Image.open(img_info["path"])
                    if image_width_mode == "Fit to Column (Side-by-Side)":
                        st.image(img, use_container_width=True)
                    else:
                        st.image(img, width=zoom_width)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
            else:
                st.warning(f"File not found: {img_info['filename']}")
else:
    st.warning("No images listed for this group.")

# Extracted QA & Evidence details expander
qa_items = current_group.get("qa_results", [])
if qa_items:
    with st.expander(f"📝 Extracted QA & Page Evidence ({len(qa_items)} QA pairs)", expanded=bool(search_query)):
        for idx, qa in enumerate(qa_items, 1):
            q_hl = highlight_html(qa.get('query', ''), search_query) if search_query else html.escape(qa.get('query', ''))
            a_hl = highlight_html(qa.get('answer', ''), search_query) if search_query else html.escape(qa.get('answer', ''))
            cat = html.escape(qa.get('category', 'General'))
            
            st.markdown(f"**Q{idx} ({cat}):** {q_hl}", unsafe_allow_html=True)
            st.markdown(f"**A{idx}:** {a_hl}", unsafe_allow_html=True)
            
            ev_list = qa.get("evidence", [])
            if ev_list:
                st.markdown("*Evidence Statements:*")
                for ev in ev_list:
                    p = ev.get('page', '?')
                    stmt = highlight_html(ev.get('statement', ''), search_query) if search_query else html.escape(ev.get('statement', ''))
                    subj = html.escape(ev.get('subject', ''))
                    st.markdown(
                        f"- **Page {p}** ({ev.get('type', 'text')}): *{subj}* — {stmt}",
                        unsafe_allow_html=True,
                    )
            st.markdown("---")

# Bottom navigation bar
st.markdown("---")
b_col1, b_col2, b_col3 = st.columns([1, 4, 1])
with b_col1:
    st.button("⬅️ Previous Group", key="bottom_prev", on_click=prev_group, use_container_width=True)
with b_col3:
    st.button("Next Group ➡️", key="bottom_next", on_click=next_group, use_container_width=True)
