# =============================================================================
# Main Streamlit Application - MF Prospectus Extraction System
# =============================================================================
#
# Front-end layer. Talks to FastAPI backend via HTTP.
# Sections are DYNAMIC – discovered by the backend's vision-based scanner.
# Document validation is performed on upload (score ≥ 65 to proceed).
#
# Start the backend first:
#   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Then run:
#   streamlit run app.py
# =============================================================================

import streamlit as st
import requests
import base64
import json
from typing import Dict, Optional
from PIL import Image
import io

# =============================================================================
# Backend API base URL
# =============================================================================

API_BASE = "http://localhost:8000"

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="MF Prospectus Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    .section-header {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .subsection-item {
        padding: 0.3rem 1rem;
        margin-left: 1rem;
        border-left: 3px solid #1f77b4;
    }
    .highlight-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .validation-pass {
        background-color: #d4edda;
        padding: 0.7rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .validation-fail {
        background-color: #f8d7da;
        padding: 0.7rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    defaults = {
        "document_processed": False,
        "current_page": 0,
        "total_pages": 0,
        "highlight_bbox": None,
        "selected_section": None,
        "selected_subsection": None,
        "chat_history": [],
        "extracted_sections": {},
        "discovered_sections": {},  # dynamic sections from scanner
        "validation_result": {},     # validation score details
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# =============================================================================
# API helpers
# =============================================================================

def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend API. Make sure it is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None


def api_post(path: str, json_body: dict = None, files: dict = None, timeout: int = 300):
    try:
        r = requests.post(
            f"{API_BASE}{path}", json=json_body, files=files, timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend API. Make sure it is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        # Special handling for validation failure (422)
        try:
            detail = e.response.json().get("detail", {})
        except Exception:
            detail = e.response.text
        if isinstance(detail, dict) and "validation" in detail:
            return {"_validation_failed": True, **detail}
        st.error(f"API error: {e.response.text}")
        return None

# =============================================================================
# Load dynamic section definitions
# =============================================================================

def load_section_definitions():
    """Fetch discovered sections from the backend."""
    if not st.session_state.discovered_sections:
        data = api_get("/sections")
        if data:
            st.session_state.discovered_sections = data
    return st.session_state.discovered_sections

# =============================================================================
# Document Processing (with validation)
# =============================================================================

def process_uploaded_document(uploaded_file) -> bool:
    """Upload the PDF to the backend. Backend validates → scans → processes."""
    try:
        with st.spinner("Validating & processing document (Vision + Docling + ChromaDB) …"):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }
            result = api_post("/upload", files=files, timeout=600)

        if result is None:
            return False

        # Check for validation failure
        if result.get("_validation_failed"):
            validation = result.get("validation", {})
            score = validation.get("score", "?")
            threshold = validation.get("threshold", 65)
            st.session_state.validation_result = validation
            st.error(f"❌ Document Rejected — not a mutual fund document")
            st.markdown(
                f'<div class="validation-fail">'
                f"Validation Score: <b>{score}</b> / 100 "
                f"(threshold: {threshold})<br/>"
                f"<small>{result.get('message', '')}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
            reasons = validation.get("reasons", [])
            for reason in reasons:
                st.caption(f"  • {reason}")
            return False

        # Validation passed → document processed
        st.session_state.document_processed = True
        st.session_state.total_pages = result.get("total_pages", 0)

        validation = result.get("validation", {})
        st.session_state.validation_result = validation

        # Load the discovered sections from API
        sec_data = api_get("/sections")
        if sec_data:
            st.session_state.discovered_sections = sec_data

        return True

    except Exception as e:
        st.error(f"Error processing document: {e}")
        return False

# =============================================================================
# Page rendering
# =============================================================================

def render_pdf_page(page_num: int, highlight_bbox: Optional[Dict] = None):
    params = {"zoom": 1.5}
    if highlight_bbox:
        params["bbox"] = json.dumps(highlight_bbox)
    data = api_get(f"/page/{page_num}", params=params)
    if data and "image_base64" in data:
        img_bytes = base64.b64decode(data["image_base64"])
        return Image.open(io.BytesIO(img_bytes))
    return None

# =============================================================================
# Section extraction
# =============================================================================

def extract_section_content(section_key: str):
    sections = st.session_state.discovered_sections
    title = sections.get(section_key, {}).get("title", section_key)
    with st.spinner(f"Extracting {title} …"):
        result = api_post("/extract-section", json_body={"section_key": section_key})
    if result:
        st.session_state.extracted_sections[section_key] = result
        # Auto-select the just-extracted section
        st.session_state.selected_section = section_key
        st.session_state.selected_subsection = 0
    return result


def extract_all_sections():
    with st.spinner("Extracting all discovered sections …"):
        result = api_post("/extract-all", timeout=600)
    if result:
        st.session_state.extracted_sections = result
        # Auto-select the first extracted section
        first_key = next(iter(result), None)
        if first_key:
            st.session_state.selected_section = first_key
            st.session_state.selected_subsection = 0
    return result

# =============================================================================
# Navigation
# =============================================================================

def navigate_to_chunk(chunk: Dict):
    page_num = chunk.get("page_number", 1)
    st.session_state.current_page = max(0, page_num - 1)
    bbox = chunk.get("bbox")
    st.session_state.highlight_bbox = bbox if bbox else None

# =============================================================================
# Chat
# =============================================================================

def send_chat_message(user_message: str):
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    result = api_post("/chat", json_body={"query": user_message})
    if result:
        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
        sources = result.get("sources", [])
    else:
        answer = "Error contacting the backend."
        sources = []
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    return answer, sources

# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Left sidebar: file upload + validation status + dynamic section nav."""
    discovered = st.session_state.discovered_sections or {}

    with st.sidebar:
        st.title("📊 MF Prospectus Analyzer")

        # ---- File upload ----
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF", type=["pdf"],
            help="Upload a Mutual Fund prospectus / factsheet PDF",
        )
        if uploaded_file:
            if st.button("Process Document", type="primary"):
                if process_uploaded_document(uploaded_file):
                    st.success("Document validated & processed!")
                    st.rerun()

        # ---- Validation badge ----
        validation = st.session_state.validation_result
        if validation:
            score = validation.get("score", 0)
            threshold = validation.get("threshold", 65)
            is_valid = validation.get("is_valid", False)
            css = "validation-pass" if is_valid else "validation-fail"
            icon = "✅" if is_valid else "❌"
            st.markdown(
                f'<div class="{css}">{icon} Validation Score: '
                f"<b>{score}</b> / 100 (threshold {threshold})</div>",
                unsafe_allow_html=True,
            )

        # ---- Dynamic section navigation ----
        if st.session_state.document_processed and discovered:
            st.divider()
            st.header("📑 Discovered Sections")
            st.caption(f"{len(discovered)} sections found by vision scan")

            for section_key, section_config in discovered.items():
                is_extracted = section_key in st.session_state.extracted_sections
                is_selected = st.session_state.selected_section == section_key
                title = section_config.get("title", section_key)
                desc = section_config.get("description", "")
                keywords = section_config.get("keywords", [])

                with st.expander(
                    f"{'✅' if is_extracted else '📁'} {title}",
                    expanded=is_selected,
                ):
                    st.caption(desc)
                    if keywords:
                        st.caption(f"🔑 {', '.join(keywords[:6])}")

                    if not is_extracted:
                        if st.button("Extract", key=f"extract_{section_key}"):
                            extract_section_content(section_key)
                            st.rerun()
                    else:
                        # View section button
                        if st.button(
                            "📋 View Section",
                            key=f"view_{section_key}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_section = section_key
                            st.session_state.selected_subsection = 0
                            # Navigate to first chunk of first subsection
                            sec_data = st.session_state.extracted_sections.get(
                                section_key, {}
                            )
                            sub_list = sec_data.get("subsections", [])
                            if sub_list and sub_list[0].get("chunks"):
                                navigate_to_chunk(sub_list[0]["chunks"][0])
                            st.rerun()

                        subs = section_config.get("subsections", [])
                        for idx, subsection in enumerate(subs):
                            sub_sel = (
                                is_selected
                                and st.session_state.selected_subsection == idx
                            )
                            if st.button(
                                f"{'→' if sub_sel else '○'} {subsection}",
                                key=f"sub_{section_key}_{idx}",
                                use_container_width=True,
                            ):
                                st.session_state.selected_section = section_key
                                st.session_state.selected_subsection = idx
                                sec_data = st.session_state.extracted_sections.get(
                                    section_key, {}
                                )
                                sub_list = sec_data.get("subsections", [])
                                if idx < len(sub_list) and sub_list[idx].get("chunks"):
                                    navigate_to_chunk(sub_list[idx]["chunks"][0])
                                st.rerun()

            st.divider()
            if st.button("🔄 Extract All Sections", use_container_width=True):
                extract_all_sections()
                st.success("All sections extracted!")
                st.rerun()


def render_document_viewer():
    st.header("📄 Document Viewer")
    if not st.session_state.document_processed:
        st.info("Upload a document to view it here.")
        return

    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col1:
        if st.button("◀ Prev"):
            st.session_state.current_page = max(0, st.session_state.current_page - 1)
            st.session_state.highlight_bbox = None
            st.rerun()
    with col2:
        page_num = st.number_input(
            "Page",
            min_value=1,
            max_value=st.session_state.total_pages,
            value=st.session_state.current_page + 1,
            key="page_input",
        )
        if page_num - 1 != st.session_state.current_page:
            st.session_state.current_page = page_num - 1
            st.session_state.highlight_bbox = None
            st.rerun()
    with col3:
        st.write(f"of {st.session_state.total_pages} pages")
    with col4:
        if st.button("Next ▶"):
            st.session_state.current_page = min(
                st.session_state.total_pages - 1,
                st.session_state.current_page + 1,
            )
            st.session_state.highlight_bbox = None
            st.rerun()

    if st.session_state.highlight_bbox:
        st.markdown(
            '<div class="highlight-info">📍 Highlighted region from selected section</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Highlight"):
            st.session_state.highlight_bbox = None
            st.rerun()

    api_page = st.session_state.current_page + 1
    img = render_pdf_page(api_page, st.session_state.highlight_bbox)
    if img:
        st.image(img, use_container_width=True)
    else:
        st.error("Could not render page.")


def render_section_content():
    st.header("📋 Section Content")

    extracted = st.session_state.extracted_sections
    if not extracted:
        st.info("No sections extracted yet. Click **Extract** or **Extract All Sections** in the sidebar.")
        return

    # Section selector dropdown so the user can pick from extracted sections
    section_keys = list(extracted.keys())
    section_labels = [extracted[k].get("title", k) for k in section_keys]

    # Determine current index
    current_key = st.session_state.selected_section
    current_idx = section_keys.index(current_key) if current_key in section_keys else 0

    chosen_label = st.selectbox(
        "Select section",
        section_labels,
        index=current_idx,
        key="section_selector",
    )
    chosen_key = section_keys[section_labels.index(chosen_label)]
    if chosen_key != st.session_state.selected_section:
        st.session_state.selected_section = chosen_key
        st.session_state.selected_subsection = 0
        st.rerun()

    section = extracted.get(chosen_key)
    if not section:
        st.warning("Section data not available.")
        return

    st.subheader(section["title"])
    st.caption(section.get("description", ""))

    with st.expander("📝 Summary", expanded=True):
        st.write(section.get("summary", ""))

    st.subheader("Subsections")
    for idx, sub in enumerate(section.get("subsections", [])):
        is_selected = st.session_state.selected_subsection == idx
        with st.expander(
            f"{'📌' if is_selected else '📄'} {sub['title']}",
            expanded=is_selected,
        ):
            st.write(sub.get("content", ""))
            chunks = sub.get("chunks", [])
            if chunks:
                st.caption(f"Sources: {len(chunks)} chunks")
                for ci, chunk in enumerate(chunks[:3]):
                    page = chunk.get("page_number", "N/A")
                    has_bbox = bool(chunk.get("bbox"))
                    btn_label = f"📍 Go to Page {page}" if has_bbox else f"Go to Page {page}"
                    if st.button(
                        btn_label,
                        key=f"goto_{chosen_key}_{idx}_{ci}_{page}",
                    ):
                        navigate_to_chunk(chunk)
                        st.rerun()


def render_chatbot():
    st.header("💬 Ask Questions")

    if not st.session_state.document_processed:
        st.info("Upload a document to start asking questions.")
        return

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for src in sources[:3]:
                            page = src.get("page_number", "N/A")
                            st.caption(f"Page {page}")
                            st.text(src["text"][:200] + "…")
                            if st.button("View", key=f"view_src_{id(src)}"):
                                navigate_to_chunk(src)
                                st.rerun()

    user_input = st.chat_input("Ask a question about the prospectus …")
    if user_input:
        with st.spinner("Thinking …"):
            send_chat_message(user_input)
        st.rerun()

    st.caption("Quick questions:")
    quick_questions = [
        "What is the fund's investment objective?",
        "What are the main risk factors?",
        "What is the expense ratio?",
        "What is the minimum investment amount?",
    ]
    cols = st.columns(2)
    for idx, question in enumerate(quick_questions):
        with cols[idx % 2]:
            if st.button(question, key=f"quick_{idx}"):
                with st.spinner("Thinking …"):
                    send_chat_message(question)
                st.rerun()


# =============================================================================
# Main
# =============================================================================

def main():
    render_sidebar()

    if st.session_state.document_processed:
        tab1, tab2, tab3 = st.tabs(["📄 Document", "📋 Sections", "💬 Chat"])
        with tab1:
            render_document_viewer()
        with tab2:
            render_section_content()
        with tab3:
            render_chatbot()
    else:
        st.title("📊 Mutual Fund Prospectus Analyzer")
        st.markdown("""
        Welcome to the **MF Prospectus Extraction System**!

        ### What's New (v2)
        - 🔍 **Smart Validation** — the system scans the first 5 pages with a
          vision model to verify the upload is a genuine MF document
          (score-based: starts at 50, needs ≥ 65 to proceed)
        - 📑 **Dynamic Sections** — sections and subsections are **discovered
          automatically** by scanning every page with GPT-4o-mini vision,
          instead of using a hardcoded list
        - 🧠 **Page Memory** — the scanner keeps a sliding window of the last
          3 pages (LangChain messages) so it can detect sections that span
          multiple pages
        - 🔑 **Keyword-Boosted RAG** — discovered keywords per section are
          injected into the retrieval query for better results

        ### Getting Started
        1. Start the backend API: `uvicorn api:app --port 8000 --reload`
        2. Upload a PDF via the sidebar
        3. Click **Process Document** — validation + section discovery + indexing
        4. Explore the dynamically discovered sections or ask questions!
        """)


if __name__ == "__main__":
    main()
