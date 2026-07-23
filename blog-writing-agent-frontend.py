from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import streamlit as st

import importlib
app = importlib.import_module("blog-writing-agent-backend").app

# -----------------------------
# Custom CSS
# -----------------------------
CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }

    .metric-card {
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-card h4 {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.55;
        margin: 0 0 0.3rem 0;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        opacity: 0.5;
        margin-top: 0.2rem;
    }

    div[data-testid="stTabs"] > div > div > button {
        font-weight: 600;
        font-size: 0.88rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=p.relative_to(images_dir.parent))
    return buf.getvalue()


def word_count(md: str) -> int:
    return len(md.split())


def reading_time(md: str) -> str:
    mins = max(1, word_count(md) // 230)
    return f"{mins} min read"


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    final_state = None
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
            if isinstance(step, dict):
                if len(step) == 1 and isinstance(next(iter(step.values())), dict):
                    final_state = next(iter(step.values()))
                else:
                    final_state = step
        if final_state is not None:
            yield ("final", final_state)
            return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
            final_state = step
        if final_state is not None:
            yield ("final", final_state)
            return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer with local image support
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    clean_md = _MD_IMG_RE.sub("", md)
    clean_md = re.sub(r"\n{3,}", "\n\n", clean_md).strip()

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last:m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()
    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            try:
                st.image(src, caption=caption or (alt or None), width="stretch")
            except Exception:
                st.caption(f"Image: {alt or src}")
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists() and img_path.stat().st_size > 5000:
                try:
                    with open(img_path, "rb") as f:
                        header = f.read(4)
                    if header[:3] == b'\xff\xd8\xff' or header[:4] == b'\x89PNG':
                        st.image(str(img_path), caption=caption or (alt or None), width="stretch")
                    else:
                        st.caption(f"Image: {alt or src}")
                except Exception:
                    st.caption(f"Image: {alt or src}")
            else:
                st.caption(f"Image: {alt or src}")
        i += 1


# -----------------------------
# Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Metric cards
# -----------------------------
def render_metric_card(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><h4>{label}</h4><div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_metrics_row(final_md: str, plan_obj, image_count: int):
    wc = word_count(final_md)
    rt = reading_time(final_md)
    task_count = 0
    if plan_obj:
        tasks = plan_obj.get("tasks", []) if isinstance(plan_obj, dict) else (plan_obj.tasks if hasattr(plan_obj, "tasks") else [])
        task_count = len(tasks)

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Words", f"{wc:,}", rt)
    with cols[1]:
        render_metric_card("Sections", str(task_count), "in outline")
    with cols[2]:
        render_metric_card("Images", str(image_count), "fetched")
    with cols[3]:
        render_metric_card("Status", "Complete", "all sections written")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Blog Writer Agent", layout="wide", page_icon="✍")

st.markdown("<h1 style='margin-bottom:0'>✍ Blog Writer Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888;margin-top:0'>AI-powered technical blog generation with research, images, and markdown export</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Generate New Blog")
    topic = st.text_area(
        "Topic",
        placeholder="Enter a topic for the blog...",
        height=100,
    )
    as_of = st.date_input("Date", value=date.today(), label_visibility="collapsed")

    btn_cols = st.columns([3, 2])
    with btn_cols[0]:
        run_btn = st.button("Generate Blog", type="primary", use_container_width=True)
    with btn_cols[1]:
        if st.button("Clear", use_container_width=True):
            st.session_state["last_out"] = None
            st.rerun()

    st.divider()
    st.markdown("### Past Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs yet.")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:30]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("Load", use_container_width=True):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.rerun()

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# -----------------------------
# Run graph
# -----------------------------
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    progress_bar = st.progress(0, text="Starting...")
    status_area = st.empty()
    steps = ["Routing", "Research", "Planning", "Writing sections", "Merging", "Finding images", "Done"]
    step_idx = 0

    current_state: Dict[str, Any] = inputs.copy()
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))

            if node_name and node_name != last_node:
                last_node = node_name
                node_to_step = {
                    "router": 0, "research": 1, "orchestrator": 2,
                    "worker": 3, "reducer": 4,
                }
                if node_name in node_to_step:
                    step_idx = node_to_step[node_name]
                elif node_name == "merge_content":
                    step_idx = 4
                elif node_name in ("decide_images", "generate_and_place_images"):
                    step_idx = 5
                pct = min((step_idx + 1) / len(steps), 1.0)
                progress_bar.progress(pct, text=f"{steps[step_idx]}...")
                status_area.caption(f"Running: **{node_name}**")

            current_state = extract_latest_state(current_state, payload)

        elif kind == "final":
            final_out = extract_latest_state(current_state, payload)
            st.session_state["last_out"] = final_out
            progress_bar.progress(1.0, text="Complete!")
            status_area.empty()

# -----------------------------
# Render results
# -----------------------------
out = st.session_state.get("last_out")
if out:
    final_md = out.get("final") or ""
    image_specs = out.get("image_specs") or []
    plan_obj = out.get("plan")

    if final_md:
        render_metrics_row(final_md, plan_obj, len(image_specs))
        st.markdown("")

    tab_preview, tab_plan = st.tabs(
        ["Preview", "Plan"]
    )

    with tab_preview:
        if not final_md:
            st.info("No output yet.")
        else:
            render_markdown_with_local_images(final_md)
            st.markdown("---")

            blog_title = "blog"
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            dl_cols = st.columns(2)
            with dl_cols[0]:
                st.download_button(
                    "Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=f"{safe_slug(blog_title)}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl_cols[1]:
                bundle = bundle_zip(final_md, f"{safe_slug(blog_title)}.md", Path("images"))
                st.download_button(
                    "Download Bundle (MD + images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    with tab_plan:
        if not plan_obj:
            st.info("No plan available.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"**{plan_dict.get('blog_title', '')}**")
            meta_cols = st.columns(3)
            meta_cols[0].markdown(f"**Audience:** {plan_dict.get('audience', '')}")
            meta_cols[1].markdown(f"**Tone:** {plan_dict.get('tone', '')}")
            meta_cols[2].markdown(f"**Type:** {plan_dict.get('blog_kind', '')}")

            tasks = plan_dict.get("tasks", [])
            if tasks:
                st.markdown("")
                for t in sorted(tasks, key=lambda x: x.get("id", 0)):
                    tags = []
                    if t.get("requires_research"):
                        tags.append("research")
                    if t.get("requires_citations"):
                        tags.append("citations")
                    if t.get("requires_code"):
                        tags.append("code")
                    tag_str = " ".join(f"`{tag}`" for tag in tags) if tags else ""

                    st.markdown(
                        f"**{t.get('id', '')}. {t.get('title', '')}** "
                        f"<span style='color:#999;font-size:0.85rem'>~{t.get('target_words', 0)} words</span> {tag_str}",
                        unsafe_allow_html=True,
                    )
                    st.caption(t.get("goal", ""))
                    st.markdown("")


else:
    st.info("Enter a topic in the sidebar and click **Generate Blog**.")
