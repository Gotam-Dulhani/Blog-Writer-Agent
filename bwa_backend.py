
from __future__ import annotations

import logging
import operator
import os
import re
import urllib.parse
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

log = logging.getLogger("bwa")

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str = ""
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. overview.jpg")
    alt: str
    caption: str
    prompt: str = Field(..., description="Search keywords for stock photo sites, e.g. 'server rack datacenter'")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


class ReducerState(TypedDict):
    topic: str
    plan: Optional[Plan]
    sections: List[tuple[int, str]]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# -----------------------------
# 2) LLM — Groq (primary) + Mistral (fallback)
# -----------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
llm_fallback = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))


def invoke_structured(pydantic_model, messages):
    """Invoke structured output: Groq first, then Mistral."""
    try:
        structured_llm = llm.with_structured_output(pydantic_model)
        return structured_llm.invoke(messages)
    except Exception as e:
        log.debug("Groq structured output failed: %s", e)
    try:
        structured_llm = llm_fallback.with_structured_output(pydantic_model, method="json_mode")
        return structured_llm.invoke(messages)
    except Exception as e:
        log.debug("Mistral structured output also failed: %s", e)
        raise

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    try:
        # Use our helper function to invoke structured output
        decision = invoke_structured(
            RouterDecision,
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
            ]
        )
    except Exception as e:
        log.debug("Structured output failed in router: %s", e)
        # Last resort fallback
        return {
            "needs_research": False,
            "mode": "closed_book",
            "queries": [],
            "recency_days": 3650,
        }

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    truncated_raw = str(raw)[:8000]

    try:
        extractor = llm.with_structured_output(EvidencePack)
        pack = extractor.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"As-of date: {state['as_of']}\n"
                        f"Recency days: {state['recency_days']}\n\n"
                        f"Raw results:\n{truncated_raw}"
                    )
                ),
            ]
        )

        dedup = {}
        for e in pack.evidence:
            if e.url:
                dedup[e.url] = e
        evidence = list(dedup.values())

        if state.get("mode") == "open_book":
            as_of = date.fromisoformat(state["as_of"])
            cutoff = as_of - timedelta(days=int(state["recency_days"]))
            evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]
    except Exception as e:
        log.debug("Error in research_node: %s", e)
        evidence = []

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and editor at a top developer blog. Create a highly actionable outline for a technical blog post.

QUALITY RULES FOR THE OUTLINE:
- 5-9 tasks. Each task title must be specific and compelling (not generic like "Introduction" or "Overview").
- Good titles sound like article headlines: "Why Your CI Pipeline Breaks at 3 AM (And How to Fix It)"
- Bad titles: "Introduction", "Conclusion", "Overview of X", "Benefits of Y"
- The goal for each task must describe what the reader will KNOW or be able to DO after reading.
- Bullets must be specific sub-topics, not vague themes. Each bullet should be answerable in 1 paragraph.
- Target words per task: 120-550. Vary lengths — deep-dive sections get 400-550, transitions get 120-200.

BLOG STRUCTURE (follow this arc):
1. Hook / Problem framing (what's broken or why this matters NOW)
2. Context / Current landscape (what people do today)
3. Deep dive 1 (core technique or tool)
4. Deep dive 2 (advanced or alternative approach)
5. Practical walkthrough / worked example
6. Gotchas / trade-offs / failure modes
7. What to do next / actionable takeaways

Make the outline read like a real blog someone would share — not a course syllabus.

Grounding:
- closed_book: evergreen concepts; no evidence dependence.
- hybrid: use evidence for current examples/tools; mark those tasks requires_research=True and requires_citations=True.
- open_book: set blog_kind='news_roundup'. Focus on events, launches, and implications.
  Do NOT invent events not supported by evidence. If evidence is sparse, reduce tasks accordingly.

Output must match Plan schema exactly.
"""

def orchestrator_node(state: State) -> dict:
    try:
        mode = state.get("mode", "closed_book")
        evidence = state.get("evidence", [])
        forced_kind = "news_roundup" if mode == "open_book" else None

        plan = invoke_structured(
            Plan,
            [
                SystemMessage(content=ORCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Mode: {mode}\n"
                        f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                        f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                        f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                    )
                ),
            ]
        )
        if forced_kind:
            plan.blog_kind = "news_roundup"
        return {"plan": plan}
    except Exception as e:
        log.debug("Structured output failed in orchestrator: %s", e)
        topic = state['topic']
        fallback_plan = Plan(
            blog_title=topic,
            audience="Developers and technical decision-makers evaluating practical approaches",
            tone="Direct, practical, and slightly opinionated",
            blog_kind="hybrid",
            constraints=["API quota exceeded - using enhanced fallback plan"],
            tasks=[
                Task(
                    id=1,
                    title="What the problem actually looks like",
                    goal=f"Frame the real pain points that make {topic} worth reading about today.",
                    bullets=["Common symptoms before adopting a solution", "What breaks when teams ignore this", "Who benefits most"],
                    target_words=250
                ),
                Task(
                    id=2,
                    title="How current approaches measure up",
                    goal=f"Survey the most common ways teams deal with {topic} today, with trade-offs.",
                    bullets=["At least 3 practical approaches or tools", "Latency, cost, or complexity trade-offs", "What to pick for small vs large scale"],
                    target_words=300
                ),
                Task(
                    id=3,
                    title="A worked example",
                    goal=f"Show a concrete walk-through so the reader leaves with actionable context.",
                    bullets=["Step-by-step reasoning", "Config/code/config changes that matter", "Expected output or observable behavior"],
                    target_words=400
                ),
                Task(
                    id=4,
                    title="Gotchas and real-world limitations",
                    goal=f"Be honest about where these approaches fail so readers can plan around them.",
                    bullets=["Edge cases that surprise practitioners", "Monitoring and debugging tips", "Common misconceptions"],
                    target_words=350
                ),
                Task(
                    id=5,
                    title="What comes next",
                    goal=f"Point the reader toward useful next steps without drifting into hype.",
                    bullets=["Short-term experiments to run this week", "Tools or standards to watch", "When to revisit the decision"],
                    target_words=300
                ),
            ]
        )
        return {"plan": fallback_plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer at a top developer-focused publication. Write ONE section of a technical blog post in Markdown.

OUTPUT FORMAT:
- Start with a level-2 heading: "## <Section Title>"
- After the heading, write a 2-4 sentence hook paragraph that draws the reader in.
- Then expand each bullet into a rich, substantive paragraph.

WRITING QUALITY (critical):
- Write like you're explaining to a smart colleague over coffee — not a textbook.
- Every paragraph must have: a concrete claim, the "why it matters", and at least one specific example (tool name, number, scenario, or analogy).
- Use transitions: "Here's the thing...", "In practice...", "What this means is..."
- BAN these phrases entirely: "It is important to note", "This deserves careful consideration", "In today's rapidly evolving", "Let's dive in", "Without further ado"
- Include real-world anecdotes, trade-offs, or "war stories" when possible.
- If a comparison is needed, use a mini table or concrete side-by-side.

DEPTH RULES:
- Expand EVERY bullet into 4-8 sentences. Do NOT just list or summarize bullets.
- Each paragraph must contain at least one of: code snippet, named tool/library, specific number/metric, or a worked scenario.
- Target word count: {target_words} words. Stay within 80%-120% of target.

SCOPE GUARD:
- If blog_kind == "news_roundup": focus on events, launches, and implications. Do NOT drift into tutorials.
- If mode == "open_book": only make claims supported by provided Evidence URLs. Cite as [Source](URL).
- If requires_citations == true: cite Evidence URLs for external factual claims.
- If requires_code == true: include 1-3 minimal, well-commented code snippets in fenced blocks.

WHAT NOT TO DO:
- Do NOT write generic introductions like "In the world of X, Y is becoming increasingly important"
- Do NOT pad with filler sentences
- Do NOT repeat the section title in the first sentence
- Do NOT use bullet lists when paragraphs would be richer
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date: unknown'}"
        for e in evidence[:20]
    )
    worker_system = WORKER_SYSTEM.replace("{target_words}", str(task.target_words))
    messages = [
        SystemMessage(content=worker_system),
        HumanMessage(
            content=(
                f"Blog title: {plan.blog_title}\n"
                f"Audience: {plan.audience}\n"
                f"Tone: {plan.tone}\n"
                f"Blog kind: {plan.blog_kind}\n"
                f"Constraints: {plan.constraints}\n"
                f"Topic: {payload['topic']}\n"
                f"Mode: {payload.get('mode')}\n"
                f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                f"Section title: {task.title}\n"
                f"Goal: {task.goal}\n"
                f"Target words: {task.target_words}\n"
                f"Tags: {task.tags}\n"
                f"requires_research: {task.requires_research}\n"
                f"requires_citations: {task.requires_citations}\n"
                f"requires_code: {task.requires_code}\n"
                f"Bullets:{bullets_text}\n\n"
                f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        log.debug("Groq failed in worker for task %s: %s", task.id, e)
        try:
            response = llm_fallback.invoke(messages)
        except Exception as e2:
            log.debug("Mistral also failed in worker for task %s: %s", task.id, e2)
            topic = payload['topic']
            section_title = task.title
            goal = task.goal
            bullets_joined = "\n\n".join(
                f"**{b.strip()}** — This is a key aspect of {topic} that practitioners encounter regularly. "
                f"Understanding how this works in practice, and what trade-offs to expect, "
                f"can save significant debugging time and lead to better architectural decisions."
                for b in task.bullets
            )

            fallback_content = (
                f"## {section_title}\n\n"
                f"{goal}\n\n"
                f"When working with {topic}, this is one of those areas where the gap between "
                f"theory and practice shows up fast. Teams often underestimate the complexity here, "
                f"only to discover edge cases in production that weren't obvious during initial evaluation.\n\n"
                f"{bullets_joined}\n\n"
                f"The key takeaway is that {topic} rewards careful attention to these details. "
                f"Start with the simplest approach that could work, measure its behavior under your actual workload, "
                f"and iterate from there based on what you observe — not what you assume."
            )
            return {"sections": [(task.id, fallback_content)]}

    # Handle case where content is a list
    if isinstance(response.content, list):
        section_md = "".join([str(part) for part in response.content]).strip()
    else:
        section_md = str(response.content).strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: ReducerState) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    # Deduplicate sections by task id
    seen_ids = set()
    unique_sections = []
    for task_id, md in state["sections"]:
        if task_id not in seen_ids:
            seen_ids.add(task_id)
            unique_sections.append((task_id, md))
    ordered_sections = [md for _, md in sorted(unique_sections, key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are a technical editor deciding where images would improve a blog post.

RULES:
- Add 1-3 images maximum. Only add an image if it genuinely helps explain a concept.
- Good image candidates: architecture diagrams, flowcharts, comparison tables, real-world screenshots.
- BAD image candidates: decorative images, stock photos that don't relate to the content.
- Place [[IMAGE_N]] placeholders on their own line, immediately AFTER the heading or paragraph where the image adds value.
- Each image needs: a clear filename, alt text, caption, and a "prompt" field with SEARCH KEYWORDS (2-5 words) to find a relevant image on stock photo sites (e.g. "cloud architecture diagram", "server monitoring dashboard", "python code terminal").
- The "prompt" field is used for IMAGE SEARCH, not AI generation. Use short, specific keywords.
- Size: use "1024x1024" for diagrams, "1536x1024" for wide charts/flowcharts.
- If the blog is purely text-based (opinion piece, news roundup), set images=[] and return the text unchanged.

OUTPUT: Return GlobalImagePlan with md_with_placeholders (the blog with [[IMAGE_N]] inserted) and images list.
"""

def _strip_images_from_md(md: str) -> str:
    """Remove markdown image syntax from text so LLMs don't interpret it as image input."""
    return re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md).strip()


def decide_images(state: ReducerState) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    clean_md = _strip_images_from_md(merged_md)
    if len(clean_md) > 24000:
        clean_md = clean_md[:24000]

    try:
        planner = llm.with_structured_output(GlobalImagePlan)
        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        "Insert placeholders + propose image prompts.\n\n"
                        f"{clean_md}"
                    )
                ),
            ]
        )
        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in image_plan.images],
        }
    except Exception as e:
        log.debug("Groq failed in decide_images: %s", e)

    try:
        planner = llm_fallback.with_structured_output(GlobalImagePlan)
        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        "Insert placeholders + propose image prompts.\n\n"
                        f"{clean_md}"
                    )
                ),
            ]
        )
        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in image_plan.images],
        }
    except Exception as e2:
        log.debug("Mistral failed in decide_images: %s", e2)
        topic = state['topic']

        lines = merged_md.split('\n')
        modified_lines = []
        placeholder_inserted = 0
        h2_count = 0

        for line in lines:
            modified_lines.append(line)
            if line.startswith('## '):
                h2_count += 1
                if h2_count in (2, 4) and placeholder_inserted < 2:
                    placeholder_num = placeholder_inserted + 1
                    modified_lines.append(f"\n[[IMAGE_{placeholder_num}]]\n")
                    placeholder_inserted += 1

        md_with_placeholders = '\n'.join(modified_lines)

        fallback_image_specs = [
                {
                    "placeholder": "[[IMAGE_1]]",
                    "filename": f"{_safe_slug(topic)}_overview.jpg",
                    "alt": f"Overview of {topic}",
                    "caption": f"A visual overview of {topic}",
                    "prompt": f"{topic} technology overview",
                    "size": "1024x1024",
                    "quality": "medium"
                },
                {
                    "placeholder": "[[IMAGE_2]]",
                    "filename": f"{_safe_slug(topic)}_details.jpg",
                    "alt": f"Key details of {topic}",
                    "caption": f"Deep dive into {topic}",
                    "prompt": f"{topic} diagram explanation",
                    "size": "1024x1024",
                    "quality": "medium"
                }
            ]

        return {
            "md_with_placeholders": md_with_placeholders,
            "image_specs": fallback_image_specs,
        }


def _search_and_fetch_image(query: str) -> bytes:
    """
    Search Pexels for a real photo matching the query, download and return bytes.
    Falls back to LoremFlickr if Pexels fails or no key is set.
    """
    import requests

    pexels_key = os.getenv("PEXELS_API_KEY")

    if pexels_key:
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": pexels_key},
                params={"query": query, "per_page": 5, "orientation": "landscape"},
                timeout=15,
            )
            resp.raise_for_status()
            photos = resp.json().get("photos", [])
            if photos:
                img_url = photos[0]["src"]["large"]
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                if len(img_resp.content) > 1000:
                    return img_resp.content
        except Exception as e:
            log.debug("Pexels search failed: %s", e)

    keywords = query.replace(",", " ").split()[:3]
    search_query = ",".join(keywords)
    try:
        url = f"https://loremflickr.com/1024/768/{search_query}"
        resp = requests.get(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) > 1000:
            return resp.content
    except Exception as e:
        log.debug("LoremFlickr fallback failed: %s", e)

    raise RuntimeError(f"Could not fetch image for: {query}")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: ReducerState) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    images_dir = Path("images")
    if images_dir.exists():
        for old_file in images_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()
    images_dir.mkdir(exist_ok=True)

    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    fetched_count = 0
    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if out_path.exists() and out_path.stat().st_size > 5000:
            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)
            fetched_count += 1
            continue

        search_query = spec.get("prompt") or spec.get("alt") or state["topic"]
        try:
            img_bytes = _search_and_fetch_image(search_query)
            if img_bytes[:3] == b'\xff\xd8\xff' or img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                out_path.write_bytes(img_bytes)
                img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
                md = md.replace(placeholder, img_md)
                fetched_count += 1
                log.debug("Fetched image: %s (%d bytes)", filename, len(img_bytes))
            else:
                log.debug("Downloaded data is not a valid image for %s", filename)
                md = md.replace(placeholder, f"> **{spec.get('alt','')}**\n> {spec.get('caption','')}\n")
        except Exception as e:
            log.debug("Image fetch failed for %s: %s", filename, e)
            search_block = (
                f"> **Image: {spec.get('alt','')}**\n"
                f"> {spec.get('caption','')}\n"
            )
            md = md.replace(placeholder, search_block)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    log.debug("Blog saved as %s (%d/%d images fetched)", filename, fetched_count, len(image_specs))
    return {"final": md}

# -----------------------------
# Reducer subgraph wrapper to prevent duplicate sections
# -----------------------------
# build reducer subgraph with ReducerState
reducer_graph = StateGraph(ReducerState)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph_compiled = reducer_graph.compile()

def reducer_node(state: State) -> dict:
    reducer_input: ReducerState = {
        "topic": state["topic"],
        "plan": state["plan"],
        "sections": state["sections"],
        "merged_md": state.get("merged_md", ""),
        "md_with_placeholders": state.get("md_with_placeholders", ""),
        "image_specs": state.get("image_specs", []),
        "final": state.get("final", "")
    }
    reducer_result = reducer_subgraph_compiled.invoke(reducer_input)
    return {
        "merged_md": reducer_result["merged_md"],
        "md_with_placeholders": reducer_result["md_with_placeholders"],
        "image_specs": reducer_result["image_specs"],
        "final": reducer_result["final"]
    }

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
