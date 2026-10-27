import json
import os
from pathlib import Path
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
                    
                    groups.append({
                        "key": key,
                        "document": doc,
                        "pages": pages,
                        "images": resolved_images,
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

groups = load_groups(selected_source)

if not groups:
    st.error(f"No groups found for {selected_source}.")
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
st.subheader(f"{current_group['document']}")
st.caption(
    f"Group **{st.session_state.group_idx + 1} of {len(groups)}** | "
    f"Group Key: `{current_group['key']}` | "
    f"Pages: **{', '.join(str(p) for p in current_group['pages'])}**"
)

# Render images side-by-side using Streamlit columns
if num_images > 0:
    cols = st.columns(num_images)
    for col, img_info in zip(cols, current_group["images"]):
        with col:
            st.markdown(f"### Page {img_info['page']}")
            st.caption(f"`{img_info['filename']}`")
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

# Bottom navigation bar
st.markdown("---")
b_col1, b_col2, b_col3 = st.columns([1, 4, 1])
with b_col1:
    st.button("⬅️ Previous Group", key="bottom_prev", on_click=prev_group, use_container_width=True)
with b_col3:
    st.button("Next Group ➡️", key="bottom_next", on_click=next_group, use_container_width=True)
