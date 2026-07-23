# Blog Writer Agent

An AI-powered multi-agent blog generation system built with LangGraph, Streamlit, and Groq.

## What it does

Give it a topic, and it researches, outlines, writes, and illustrates a full technical blog post in Markdown — with real stock images and a one-click ZIP download.

## Architecture

```
Topic → Router → (Research?) → Planner → Workers (parallel) → Merger → Image Planner → Fetcher → Blog
```

| Node | Role |
|---|---|
| **Router** | Decides if web research is needed (closed_book / hybrid / open_book) |
| **Research** | Pulls recent sources via Tavily API |
| **Planner** | Generates a structured outline with audience, tone, and section goals |
| **Workers** | Each writes one section in parallel (Groq Llama 3.3 70B, Mistral fallback) |
| **Merger** | Orders sections and concatenates into a single Markdown document |
| **Image Planner** | Picks 1–3 spots for images and generates search keywords |
| **Image Fetcher** | Downloads real photos from Pexels (or LoremFlickr fallback) |

## Stack

- **LangGraph** — multi-agent orchestration with parallel fan-out
- **Groq (Llama 3.3 70B)** — primary LLM for all generation tasks
- **Mistral Large** — automatic fallback if Groq is unavailable
- **Tavily** — web research for open/hybrid topics
- **Pexels / LoremFlickr** — real stock images
- **Streamlit** — minimal dark-themed UI

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/bwa.git
cd bwa
python -m venv myvenv
myvenv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
TAVILY_API_KEY=tvly-...
PEXELS_API_KEY=...          # optional, for better images
```

```bash
streamlit run blog-writing-agent-frontend.py
```

## Example Output

Enter any topic — e.g. *"Rust vs Go for backend services"* — and the agent will:

1. Route to open_book mode (current tooling matters)
2. Research recent comparisons and benchmarks
3. Plan 5–7 sections with specific goals
4. Write each section with code examples and citations
5. Fetch relevant images and place them in the Markdown
6. Save the blog as `topic_name.md` in the project root

## License

MIT
